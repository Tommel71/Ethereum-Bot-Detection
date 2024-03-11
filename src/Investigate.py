import time
from tools import load_configs
from src.Aggregate import Aggregate
from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import os
import plotly.graph_objs as go
import datetime
from src.Preprocessing.FeaturesProcessor import FeaturesModeller
from src.Analysis.benchmark_clustering import benchmark, eval_clustering
from src.Analysis.benchmark_heuristics import accuracy_prediction
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from tools import save_data_for_figure
import matplotlib.pyplot as plt
from src.Datamodels.PipelineComponent import PipelineComponent
from tools import save_table, psql_insert_df, save_prediction_results, load_prediction_results, log_time, save_data
from functools import partial
import scipy.stats as st
from xgboost import XGBClassifier
from tools import save_data
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from tools import load_data
import bz2
import pickle
import _pickle as cPickle
from src.Analysis.benchmark_clustering import purity_score, entropy_score
from sklearn.impute import SimpleImputer
from src.Analysis.benchmark_clustering import purity_scores_single, entropy_scores_single
from time import time

# create a dictionary of scores, that reports the mean and standard deviation of the scores
# get a 95% confidence interval for the mean
def get_ci(x):
    int = st.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=st.sem(x))
    int_rounded = [f"{i:.2f}" for i in int]
    return tuple(int_rounded)


reportstring = lambda x: f"{x.mean():.2f} {get_ci(x)}".replace("'", "")


def classification_metrics_from_binary_predictions(y_test, y_pred, with_ci=False):

    if not with_ci:
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "N Positives": np.sum(y_pred),
        }

    # create 100 bootstrap samples and calculate scores. Then report with 95% confidence interval
    metrics = []
    for i in range(100):
        idx = np.random.choice(len(y_test), len(y_test), replace=True)
        metrics.append(
            {
                "Accuracy": accuracy_score(y_test[idx], y_pred[idx]),
                "Precision": precision_score(y_test[idx], y_pred[idx]),
                "Recall": recall_score(y_test[idx], y_pred[idx]),
                "F1": f1_score(y_test[idx], y_pred[idx]),
            }
        )

    # calculate mean and confidence interval
    cis = {
        k: get_ci([r[k] for r in metrics])
        for k in ["Accuracy", "Precision", "Recall", "F1"]
    }

    means = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    }

    result = {
        "Accuracy": f"{means['Accuracy']:.2f} {cis['Accuracy']}",
        "Precision": f"{means['Precision']:.2f} {cis['Precision']}",
        "Recall": f"{means['Recall']:.2f} {cis['Recall']}",
        "F1": f"{means['F1']:.2f} {cis['F1']}",
        "N Positives": str(sum(y_pred)),
    }
    result = {k: v.replace("'", "") for k, v in result.items()}

    return result


def get_scores(pipeline, X, y, classes, multiclass=False, negative_classes=("human",)):
    # Perform cross validation with a random forest classifier within each fold of KFold
    cv = KFold(n_splits=20, shuffle=True, random_state=0)

    data = []
    preds = []
    indices = []

    # map y to numeric if string
    str_labels = False
    if isinstance(y[0], str):
        str_labels = True
        unique_y = np.unique(y)
        mapping = {c: i for i, c in enumerate(unique_y)}
        y = np.array([mapping[c] for c in y])
        reverse_mapping = {i: c for c, i in mapping.items()}

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        try:
            y_pred = pipeline.predict(X_test)
        except:
            print("ad")
        if multiclass:
            data += [
                {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="macro"),
                    "Recall": recall_score(y_test, y_pred, average="macro"),
                    "F1": f1_score(y_test, y_pred, average="macro"),

                }
            ]
        else:
            data += [
                {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred),
                    "Recall": recall_score(y_test, y_pred),
                    "F1": f1_score(y_test, y_pred),
                }
            ]


        #else:
        #    data += [
        #        {
        #            "Accuracy": accuracy_score(y_test, y_pred),
        #            "Precision": precision_score(y_test, y_pred),
        #            "Recall": recall_score(y_test, y_pred),
        #            "F1": f1_score(y_test, y_pred),
        #        }
        #    ]

        preds += [y_pred]
        indices += [test_idx]

    y_pred_full = np.concatenate(preds)
    ordering = np.concatenate(indices)
    y_pred_ordered = y_pred_full[np.argsort(ordering)]

    # map back
    if str_labels:
        y_pred_ordered = np.array([reverse_mapping[i] for i in y_pred_ordered])

    category_matrix = pd.crosstab(y_pred_ordered,
                                  classes).T

    category_matrix_only = category_matrix.copy()

    if multiclass:

        category_matrix["correct"] = [category_matrix_only.loc[category, category] for category in
                                      category_matrix_only.index]
        category_matrix["Accuracy"] = category_matrix["correct"] / category_matrix_only.sum(axis=1)
        category_matrix["N Samples"] = category_matrix_only.sum(axis=1)


    else:
        category_matrix["correct"] = category_matrix[1]
        category_matrix["Accuracy"] = category_matrix["correct"] / category_matrix[[0, 1]].sum(axis=1)
        category_matrix["N Samples"] = category_matrix[[0, 1]].sum(axis=1)

    # for each category, get if the address is a positive subclass or not and invert
    for category in category_matrix.index:
        if category in negative_classes:
            category_matrix.loc[category, "Accuracy"] = 1 - category_matrix.loc[category, "Accuracy"]

    category_matrix.drop(columns=["correct"], inplace=True)

    detailed = category_matrix[["Accuracy", "N Samples"]].sort_values("N Samples", ascending=False).T

    df = pd.DataFrame(data)



    scores = {
        "Accuracy": reportstring(df["Accuracy"]),
        "Precision": reportstring(df["Precision"]),
        "Recall": reportstring(df["Recall"]),
        "F1": reportstring(df["F1"]),
    }

    return scores, detailed, category_matrix_only

class Analysis(PipelineComponent):

    def __init__(self, configs, prefix):
        super().__init__(configs, prefix)

        self.agg = Aggregate(configs, prefix=prefix)
        self.fp = FeaturesModeller(configs, prefix=prefix, nafill_type=configs["Heuristics"]["nafill"])


    def color_palette_for_set(self, set):
        set_l = list(set)
        set_size = len(set_l)
        cmap = plt.cm.get_cmap('tab10', set_size).colors
        mapping = {set_l[i]: f"rgb({cmap[i][0]}, {cmap[i][1]}, {cmap[i][2]})" for i in range(set_size)}
        return mapping

    def calculate_correlations(self, df):
        corr = df.corr()
        corr = corr.dropna(how="all", axis=1)
        corr = corr.dropna(how="all", axis=0)
        corr = corr.fillna(0)
        corr.to_csv("corr.csv")


    def create_dashboard(self, embedding_df, df_normalized, additional_categories_og:pd.DataFrame, port=8050):

        address_to_index = {address: i for i, address in enumerate(df_normalized.index)}
        feature_to_highlight = "tx__custom__n_tx_per_block"
        min_val = df_normalized[feature_to_highlight].min()
        mask = df_normalized[feature_to_highlight] == min_val
        embedding_df = embedding_df.copy()
        embedding_dims = len(embedding_df.columns)
        embed_red = embedding_df[mask]
        embed_blue = embedding_df[~mask]
        plt.scatter(embed_blue["x"], embed_blue["y"], c="blue")
        plt.scatter(embed_red["x"], embed_red["y"], c="red")
        # color
        plt.show()

        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

        app = Dash(__name__, external_stylesheets=external_stylesheets)

        styles = {
            'pre': {
                'border': 'thin lightgrey solid',
                'overflowX': 'scroll'
            }
        }

        embedding_df["customdata"] = range(len(embedding_df))

        scale = "Bluered"
        fig = px.scatter(embedding_df, x="x", y="y", custom_data=["customdata"], color_continuous_scale=scale, height=800,
                         width=800)
        fig.update_layout(clickmode='event+select')
        fig.update_traces(marker_size=2)

        percentage_mapping = {i: x for i, x in enumerate([0.001, 0.01, 0.1, 0.05, 1])}
        color_factor_mapping = {i: x for i, x in enumerate([1, 10, 100] + [i * 1000 for i in range(1, 10)])}
        x_lim = fig.layout.xaxis.range
        y_lim = fig.layout.yaxis.range
        app.layout = html.Div(

        [html.Div(className='row', children=[
            html.Div(className='five columns', children=[

                dcc.Dropdown(
                    df_normalized.columns,
                    id='crossfilter-column-color',
                    value='tx__custom__n_tx_per_block'
                ),

                html.Div(dcc.Slider(
                    0,
                    len(color_factor_mapping) - 1,
                    step=None,
                    id='crossfilter-year--slider',
                    value=0,
                    marks={str(i): str(color_factor_mapping[i]) for i in list(range(len(color_factor_mapping)))}
                ), style={'padding': '0px 20px 20px 20px'}),

                dcc.Dropdown(
                    df_normalized.columns,
                    id='crossfilter-column-value',
                    value='tx__custom__n_tx_per_block'
                ),
                dcc.Dropdown(
                    ["top-percentage", "bottom-percentage"],
                    id='crossfilter-top_or_bottom',
                    value='top-percentage'
                ),

                html.Div(dcc.Slider(
                    0,
                    len(percentage_mapping) - 1,
                    step=1,
                    id='crossfilter-value-slider',
                    value=4,
                    marks={str(i): str(percentage_mapping[i]) for i in list(range(len(percentage_mapping)))}
                ), style={'padding': '0px 20px 20px 20px'}),

            ]),

        ]),

         dcc.Graph(
             id='basic-interactions',
             figure=fig
         ),

         html.Div(className='row', children=[

             html.Div([
                 dcc.Markdown("""
                    Selection Data
                """),
                 html.Pre(id='selected-data', style=styles['pre']),
             ], className='three columns'),

         ])
         ])

        @callback(
            Output('basic-interactions', 'figure'),
            Input('crossfilter-year--slider', 'value'),
            Input('crossfilter-column-color', 'value'),
            Input('crossfilter-value-slider', 'value'),
            Input('crossfilter-column-value', 'value'),
            Input('crossfilter-top_or_bottom', 'value'),
        )
        def update_figure(color_scaling_ind, color_col, threshold_ind, threshold_col, top_or_bottom):

            print(color_col)
            embedding_df_to_plot = embedding_df.copy()
            embedding_df_to_plot["address"] = df_normalized.index

            embedding_df_to_plot["color"] = df_normalized[color_col].values
            embedding_df_to_plot.dropna(inplace=True)
            threshold = percentage_mapping[threshold_ind]
            print(threshold)
            if threshold < 1:
                # get the top or bottom rows from df_normalized based on threshold_col and top_or_bottom and threshold
                if top_or_bottom == "top-percentage":
                    mask = df_normalized[threshold_col] > df_normalized[threshold_col].quantile(1 - threshold)
                elif top_or_bottom == "bottom-percentage":
                    mask = df_normalized[threshold_col] < df_normalized[threshold_col].quantile(threshold)
                else:
                    raise ValueError("top_or_bottom must be either 'top-percentage' or 'bottom-percentage'")
                # adapt mask so it fits the embedding_df_to_plot
                mask = mask[embedding_df_to_plot.index].values
                print(mask)
                print(embedding_df_to_plot)
                embedding_df_to_plot = embedding_df_to_plot[mask]



            # embedding_df_to_plot["color"] = color_factor_mapping[color_scaling_ind] * embedding_df_to_plot["color"]
            # embedding_df_to_plot = embedding_df_to_plot[embedding_df_to_plot["color"] < 1]
            embedding_df_to_plot["color"] = np.log(embedding_df_to_plot["color"] + 1)
            # get the ranks of the values
            embedding_df_to_plot["color"] = embedding_df_to_plot["color"].rank(pct=True)
            # subtract min and divide by max minus min
            embedding_df_to_plot["color"] = (embedding_df_to_plot["color"] - embedding_df_to_plot["color"].min()) / (embedding_df_to_plot["color"].max() - embedding_df_to_plot["color"].min())


            if additional_categories_og is not None:
                additional_categories = additional_categories_og.copy()
                additional_color_mapping = self.color_palette_for_set(set(additional_categories["category"]))
                additional_categories["color"] = additional_categories["category"].apply(lambda x: additional_color_mapping[x])
                additional_categories.index = additional_categories.index.map(address_to_index)
                additional_categories_cut = additional_categories[additional_categories.index.isin(embedding_df_to_plot.index)]
                embedding_df_to_plot_cont = embedding_df_to_plot[~embedding_df_to_plot.index.isin(additional_categories_cut.index)]
            else:
                embedding_df_to_plot_cont = embedding_df_to_plot.copy()

            if embedding_dims == 2:
                fig = px.scatter(embedding_df_to_plot_cont, x="x", y="y", custom_data=["customdata"], color="color",color_continuous_scale="Bluered",
                                 height=800, width=1200, range_x=x_lim, range_y=y_lim, opacity=0.5, hover_data=["address"])
            if embedding_dims == 3:
                fig = px.scatter_3d(embedding_df_to_plot_cont, x="x", y="y", z="z", custom_data=["customdata"], color="color",color_continuous_scale="Bluered",
                                 height=800, width=1200, range_x=x_lim, range_y=y_lim, opacity=0.5, hover_data=["address"])

            fig.update_traces(marker_size=1 + 300 / np.sqrt(len(embedding_df_to_plot)))
            fig.update(layout_coloraxis_showscale=False)
            fig.update_layout(clickmode='event+select')

            if additional_categories_og is None:
                return fig

            for name, color in additional_color_mapping.items():
                color_df_ind = additional_categories_cut.index[(additional_categories_cut["color"] == color)]
                color_df = embedding_df_to_plot.loc[color_df_ind]

                # Add scatter trace with medium sized markers
                if embedding_dims == 2:
                    fig.add_trace(
                        go.Scatter(
                            mode='markers',
                            name=name,
                            x=color_df["x"].values,
                            y=color_df["y"].values,
                            hovertext=color_df["address"].values,
                            customdata=color_df["customdata"].values,
                            marker=dict(
                                color=color,
                                symbol="diamond",
                                size = 2*(1 + 300 / np.sqrt(len(embedding_df_to_plot))),

                            ),
                            showlegend=True,
                        )
                    )
                if embedding_dims == 3:
                    fig.add_trace(
                        go.Scatter3d(
                            mode='markers',
                            name=name,
                            x=color_df["x"].values,
                            y=color_df["y"].values,
                            z=color_df["z"].values,
                            hovertext=color_df["address"].values,
                            customdata=color_df["customdata"].values,
                            marker=dict(
                                color=color,
                                symbol="diamond",
                                size = 2*(1 + 300 / np.sqrt(len(embedding_df_to_plot))),

                            ),
                            showlegend=True,
                        )
                    )



            return fig

        @callback(
            Output('selected-data', 'children'),
            Input('basic-interactions', 'selectedData'))
        def display_selected_data(selectedData):
            if selectedData is None:
                return "No data selected"
            print(selectedData)
            rows = [selectedData['points'][i]['customdata'] if type(selectedData['points'][i]['customdata']) == int else selectedData['points'][i]['customdata'][0]  for i in range(len(selectedData['points']))]
            df = df_normalized.iloc[rows]
            ind = 0
            while os.path.exists(f"selected_{ind}.csv"):
                ind += 1

            df.to_csv(f"selected_{ind}.csv")
            etherscan =lambda x: f"https://etherscan.io/address/{x}"
            df.index = df.index.map(etherscan)

            return str(df.index)

        app.run(debug=False, port=port)


    def visualize_gmm(self, df_clustered, embedding_df):
        plt.scatter(embedding_df["x"], ["y"], c=df_clustered["cluster"])
        plt.show()

    def load_data(self, agg, fp):

        # load data
        df_features = agg.load_features(preprocessing_function=self.fp.preprocess)
        evalset = fp.load_eval_set()
        mask = np.array([address in evalset.index for address in df_features.index])
        # split data
        df_eval = df_features[mask]
        df_features = df_features[~mask]
        # normalize
        df_normalized = fp.fit_transform_normalize_df(df_features)
        df_eval_normalized = fp.transform_normalize_df(df_eval)
        return df_features, df_normalized, df_eval_normalized, evalset

    @log_time
    def embed(self, prefix, configs, df_normalized, df_eval_normalized,nafill, cache=True, dimensions=2):
        start = time()

        run = configs["General"]["run_name"]
        df_path = f"{prefix}/data/df_embedding_{run}_{nafill}_{dimensions}.csv"
        df_eval_path = f"{prefix}/data/df_embedding_evalset_{run}_{nafill}_{dimensions}.csv"
        if os.path.exists(df_path) and os.path.exists(df_eval_path) and cache:
            df_embedding = pd.read_csv(df_path, index_col=0)
            df_embedding_evalset = pd.read_csv(df_eval_path, index_col=0)
            return df_embedding, df_embedding_evalset


        # get a 50_000 row subset of the normalize df for clustering
        self.fp.fit_transform_UMAP_embedding(df_normalized.sample(min(50_000, len(df_normalized))), dimensions=dimensions)

        df_embedding_evalset = self.fp.transform_UMAP_embedding(df_eval_normalized)
        df_embedding = self.fp.transform_UMAP_embedding(df_normalized)

        if cache:
            df_embedding.to_csv(df_path)
            df_embedding_evalset.to_csv(df_eval_path)

        print("embedded", datetime.datetime.now().strftime("%H:%M:%S"))
        print(f"embedding took {time() - start} seconds")
        return df_embedding, df_embedding_evalset

    def cluster(self, df_normalized, df_eval_normalized):

        self.fp.fit_transform_BIC_optimal_GMM(df_normalized)
        print("clustered", datetime.datetime.now().strftime("%H:%M:%S"))
        evalset_clustered = self.fp.transform_BIC_optimal_GMM(df_eval_normalized)
        df_clustered = self.fp.transform_BIC_optimal_GMM(df_normalized)
        return df_clustered, evalset_clustered

    def evaluate_clustering_from_dfs(self, evalset:pd.DataFrame, evalset_clustered: pd.DataFrame):

        df_evalset_categorised = pd.DataFrame(evalset_clustered, index=evalset.index)
        df_evalset_categorised.columns = ["cluster_id"]
        df_target = pd.DataFrame(evalset["Bot"], index=evalset.index)
        df_target.columns = ["label"]
        interval, mean, scores = benchmark(df_evalset_categorised, df_target)
        accuracy, recall, precision, f1, assigned = eval_clustering(df_evalset_categorised, df_target)

        # scores is a list of measurements. We want to see if accuracy is significantly larger than values in "scores"
        # we do this by calculating the p-value
        p_value = sum([accuracy < score for score in scores]) / len(scores)
        purity = purity_score(df_target["label"], df_evalset_categorised["cluster_id"])
        entropy = entropy_score(df_target["label"], df_evalset_categorised["cluster_id"])
        print(f"accuracy: {accuracy}, recall: {recall}, precision: {precision}, f1: {f1}, p-value: {p_value}")
        return {"Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1": f1,
                "Purity":purity, "Entropy":entropy,
                "BM-mean": mean, "p-value": p_value}, \
            assigned


    def evaluate_clustering_from_dfs_light(self,evalset:pd.DataFrame, evalset_clustered: pd.DataFrame):
        df_evalset_categorised = pd.DataFrame(evalset_clustered, index=evalset.index)
        df_evalset_categorised.columns = ["cluster_id"]
        df_target = pd.DataFrame(evalset["Bot"], index=evalset.index)
        df_target.columns = ["label"]
        purity = purity_score(df_target["label"], df_evalset_categorised["cluster_id"])
        entropy = entropy_score(df_target["label"], df_evalset_categorised["cluster_id"])
        return {"Purity":purity, "Entropy":entropy}

    @log_time
    def graph_based_diamond_pattern(self):
        df_transfers = self.agg.load_transfers()
        print("duplicates", df_transfers.duplicated().sum())

        # create graph from transfers
        # first map all the addresses to numbers because thats more efficient
        addresses = set(df_transfers["from_address"].unique()).union(set(df_transfers["to_address"].unique()))
        addresses = {address: i for i, address in enumerate(addresses)}
        df_transfers["from_address"] = df_transfers["from_address"].map(addresses)
        df_transfers["to_address"] = df_transfers["to_address"].map(addresses)

        print("n transfers", len(df_transfers))
        # create graph
        n_paths = 10 + 1 # +1 because the sender is included the way we calculate. We remove it at the end

        df_min_n_out = df_transfers.groupby("from_address").agg({"to_address": "count"})
        df_min_n_in = df_transfers.groupby("to_address").agg({"from_address": "count"})
        print("aggregated")
        mask_out = df_min_n_out["to_address"] > n_paths
        mask_in = df_min_n_in["from_address"] > n_paths
        df_min_n_out = df_min_n_out[mask_out]
        df_min_n_in = df_min_n_in[mask_in]

        print("filtered aggregated")
        df_transfers_min_n_out = df_transfers[df_transfers["from_address"].isin(df_min_n_out.index)]
        df_transfers_min_n_in = df_transfers[df_transfers["to_address"].isin(df_min_n_in.index)]
        print("filtered transfers")
        print(f"lengths, min_n_out: {len(df_transfers_min_n_out)}, min_n_in: {len(df_transfers_min_n_in)}")
        print("We are not interested in lines of addresses that send x or more transactions but dont have targets that receive x or more transactions.")
        df_transfers_min_n_out_refined = df_transfers_min_n_out[df_transfers_min_n_out['to_address'].isin(df_transfers_min_n_in['from_address'])]
        df_transfers_min_n_in_refined = df_transfers_min_n_in[df_transfers_min_n_in['from_address'].isin(df_transfers_min_n_out['to_address'])]
        print(f"refined lengths, min_n_out: {len(df_transfers_min_n_out_refined)}, min_n_in: {len(df_transfers_min_n_in_refined)}")
        df_transfers_min_n_out, df_transfers_min_n_in = df_transfers_min_n_out_refined, df_transfers_min_n_in_refined

        df_left = df_transfers_min_n_out_refined
        df_right = df_transfers_min_n_in_refined

        # Perform the join on the common columns 'to_address' and 'from_address'
        result_df = df_left.merge(df_right, left_on='to_address', right_on='from_address', suffixes=('_left', '_right'))

        print("prefilter grouping", time())
        # Group by 'from_address_A' and 'to_address_B', and count the occurrences
        result_df_reduced = result_df[["from_address_left", "from_address_right", "to_address_right"]]
        grouped_prefilter_counts = result_df_reduced.groupby(['from_address_left', 'to_address_right']).size().reset_index(name='count')

        print("prefilter", time())
        # Filter for groups with more than 10 tuples
        filtered_result = grouped_prefilter_counts[grouped_prefilter_counts['count'] > n_paths]
        prefiltered = result_df[result_df['from_address_left'].isin(filtered_result['from_address_left']) & result_df['to_address_right'].isin(filtered_result['to_address_right'])]

        print("group", time())
        grouped_result = prefiltered.groupby(['from_address_left', 'to_address_right']).apply(np.array)
        grouped_result = pd.DataFrame(grouped_result, columns=["middle_addresses"])
        print("list done", time())
        grouped_result['count'] = grouped_result['middle_addresses'].apply(len)
        print("count done", time())



        # Filter for groups with more than 10 tuples
        filtered_result = grouped_result[grouped_result['count'] > n_paths]
        a = filtered_result.sort_values(by="count", ascending=False).head(10)
        filtered_result.reset_index(inplace=True)
        print("filtered", time())
        # map back the addresses
        addresses = {v: k for k, v in addresses.items()}
        filtered_result["from_address_left"] = filtered_result["from_address_left"].map(addresses)
        filtered_result["to_address_right"] = filtered_result["to_address_right"].map(addresses)
        filtered_result["middle_addresses"] = filtered_result["middle_addresses"].apply(lambda y: [addresses[i] for x in y for i in x][1:])# the first one is always the from address

        receivers = list(set(filtered_result["to_address_right"].unique()))
        suppliers = list(set(filtered_result["from_address_left"].unique()).difference(receivers))
        middle_addresses = list(set([x for y in filtered_result["middle_addresses"] for x in y]).difference(receivers).difference(suppliers))
        df_categories = pd.concat([pd.DataFrame(columns= ["address", "category"], data=[[x, "supplier"] for x in suppliers]),
                        pd.DataFrame(columns= ["address", "category"], data=[[x, "receiver"] for x in receivers]),
                        pd.DataFrame(columns= ["address", "category"], data=[[x, "middle"] for x in middle_addresses])])
        df_categories.set_index("address", inplace=True)
        return df_categories

    @log_time
    def graph_based_deposit_receiver_supplier(self, df_features, configs):

        df_transfers = self.agg.load_transfers()
        print("duplicates", df_transfers.duplicated().sum())

        #n_tx = (configs["Metadata"]["n_blocks"] * df_features["tx__custom__n_tx_per_block"]).fillna(0).astype(int)

        # extend n_tx by 0s for all the addresses in df_transfers that are not in df_features
        #addresses_not_in_features = pd.Series(0, index=(set(df_transfers["to_address"]).union(set(df_transfers["from_address"]))).difference(set(df_features.index)))
        #print("In a final run, this should be 0:", len(addresses_not_in_features))

        df_transfers["from_address"] = df_transfers["from_address"].astype("category")
        df_transfers["to_address"] = df_transfers["to_address"].astype("category")

        # Step 1: Identify addresses that send to very few addresses and receive from very few addresses
        max_transfers = configs["Heuristics"]["Deposit"]["max_transfers_deposit"]
        potential_deposit_wallets = df_transfers["from_address"].value_counts()[df_transfers["from_address"].value_counts() < max_transfers].index.union(
            df_transfers["to_address"].value_counts()[df_transfers["to_address"].value_counts() < max_transfers].index)

        # Step 2: Ensure that the address has at least one incoming and one outgoing transfer
        df_potential_deposit_wallets = df_transfers[df_transfers["from_address"].isin(potential_deposit_wallets) | df_transfers["to_address"].isin(potential_deposit_wallets)]
        # NOTE: INSTEAD OF UNION IN THE NEXT LINE, INTERSECTION WOULD BE CORRECT. A DEPOSIT WALLET RECEIVES FUNDS AND SENDS THEM TO AN AGGREGATOR WALLET
        # BUT SINCE OUR OBSERVATION WINDOW IS ONLY 100k BLOCKS WE MIGHT MISS THE INCOMING TX TO THE DEPOSIT WALLET.
        deposit_wallets = df_potential_deposit_wallets["from_address"].value_counts()[df_potential_deposit_wallets["from_address"].value_counts() > 0].index.union(
            df_potential_deposit_wallets["to_address"].value_counts()[df_potential_deposit_wallets["to_address"].value_counts() > 0].index)

        # Step 3: Select addresses that send to one of the top 1% most active addresses regarding incoming tx
        x = configs["Heuristics"]["Deposit"]["top_percentage_receiver"]
        unique_wallets_incoming = df_transfers["to_address"].unique()
        # get the top x% of the addresses with the most incoming transfers
        top_x_percent_incoming = df_transfers["to_address"].value_counts().head(int(len(unique_wallets_incoming) * x))

        # Step 4: Check if the deposit wallets have at least one transfer to a top 1% active address
        deposit_wallets_first_run = df_transfers[df_transfers["from_address"].isin(deposit_wallets) & df_transfers["to_address"].isin(top_x_percent_incoming.index)]["from_address"].unique()


        # Step 5: Find addresses that receive many incoming transactions from deposit wallets
        threshold = configs["Metadata"]["n_blocks"] / configs["Heuristics"]["Deposit"]["receiver_average_blocks_per_tx_threshold"]
        incoming_counts = df_transfers[df_transfers["from_address"].isin(deposit_wallets_first_run)]["to_address"].value_counts()
        deposit_wallet_receivers = incoming_counts[incoming_counts > threshold].index

        #incoming_counts.to_csv("incoming_counts.csv")

        # Step 6: Refine the list of deposit wallets by removing those that do not send to a receiver wallet
        df_transfers_deposit_senders = df_transfers[df_transfers["from_address"].isin(deposit_wallets_first_run) & df_transfers["to_address"].isin(deposit_wallet_receivers)]
        deposit_wallets_second_run = df_transfers_deposit_senders["from_address"].unique()

        # Step 7: Count how often each address sends to a deposit wallet
        df_transfers_deposit_receivers = df_transfers[df_transfers["to_address"].isin(deposit_wallets_second_run)]
        outgoing_counts = df_transfers_deposit_receivers["from_address"].value_counts()

        plotting = False
        if plotting:
            plt.show()
            # bar plot: color the bars red that are above the threshold
            incoming_counts[:100].plot.bar(
                color=["red" if count > threshold else "blue" for count in incoming_counts[:100]])
            plt.xticks([])  # Hide x-axis labels
            plt.ylabel("Incoming Transactions from Deposit Wallets")
            plt.title("Top 100 Deposit Wallet Recipients")
            plt.show()

            outgoing_counts[:100].plot.bar(
                color=["red" if count > threshold else "blue" for count in outgoing_counts[:100]])
            plt.xticks([])  # Hide x-axis labels
            plt.ylabel("Outgoing Transactions to Deposit Wallets")
            plt.title("Top 100 Deposit Wallet Senders")
            plt.show()

        #outgoing_counts.to_csv("outgoing_counts.csv")
        deposit_wallet_suppliers = outgoing_counts[outgoing_counts > threshold].index

        additional_categories = pd.concat([
            pd.DataFrame(len(deposit_wallet_receivers)* ["deposit_wallet_receiver"], index=deposit_wallet_receivers),
            pd.DataFrame(len(deposit_wallet_suppliers)* ["deposit_wallet_supplier"], index=deposit_wallet_suppliers),
            pd.DataFrame(len(deposit_wallets_second_run) * ["deposit_wallet"], index=deposit_wallets_second_run)
        ])
        additional_categories.columns = ["category"]
        additional_categories.reset_index(inplace=True)

        additional_categories.drop_duplicates(inplace=True, keep="first", subset="index")
        additional_categories.set_index("index", inplace=True)

        return additional_categories

    def unify_categories(self, df_list: list):
        """
        Get a list of pandas dataframes, that have addresses as indices and exactly one column "category".
        The output is a single dataframe with exactly one column called "category".
        If there are multiple categories for one address, the category with the highest priority is chosen.
        The priority is determined by the order of the input list. The first dataframe has the highest priority.
        :param df_list:
        :return:
        """
        # Combine the DataFrames using pandas.concat with axis=0 to stack them vertically.
        combined_df = pd.concat(df_list)
        combined_df.reset_index(inplace=True)

        # sort, so nans are at the end
        combined_df.sort_values(by="category", inplace=True)

        # Drop duplicates, keeping the first occurrence (highest priority) for each address.
        unified_df = combined_df.drop_duplicates(subset=["index"], keep='first')

        # Set the index to be the "address" column.
        unified_df.set_index("index", inplace=True)

        return unified_df

    def threshold_based(self, df_features):


        # a category is defined by a tuple of features with a tuple of ranges
        # e.g. category "threshold_benfords_tx_value" : (("tx__value__benfords", "tx__value__tvc"), ((0.5, 0.6), (0.5, 0.6)))
        categories = {
             "T1": (("tx__custom__n_tx_per_block", ), ((1/5,  np.inf), ))
            ,"T2": (("scb__eb__transfer__value__tvc", "tx__custom__n_tx_per_block"), ((0, 0.1), (1/100, np.inf))) # round/nonround
            ,"T3": (("tx__value__tvc", "tx__custom__n_tx_per_block"), ((0, 0.1), (1/100, np.inf))) # round/nonround
            ,"T4": (("tx__time__intime_sleepiness", ), ((0, 1*60*60), ))
            ,"T5": (("tx__time__outtime_sleepiness", ), ((0, 6*60*60), ))
            ,"T6": (("tx__time__intime_transaction_frequency", "tx__custom__n_tx_per_block"), ((0.015, np.inf), (10/self.configs["Metadata"]["n_blocks"], np.inf)))
            ,"T7": (("tx__time__outtime_transaction_frequency", "tx__custom__n_tx_per_block"), ((0.01, np.inf), (10/self.configs["Metadata"]["n_blocks"], np.inf)))
            ,
        }

        def disp(x):
            if x == np.inf:
                return "$\infty$"
            else:
                return x

        #naming_short = [f"t{i}" for i in range(1,1+len(categories))]

        description = [
            " \& ".join([f"{disp(categories[key][1][i][0])} $\le$ {categories[key][0][i]} $\le$ {disp(categories[key][1][i][1])}" for i in range(len(categories[key][0]))]) for key in categories.keys()
        ]
        df_desc = pd.DataFrame({"Short": categories.keys(), "Description": description})

        category_data = {}
        # for each category, get the addresses that are in the category
        for category, (features, ranges) in categories.items():
            # get the addresses that are in the category
            addresses_in_category = []
            for feature, range_ in zip(features, ranges):
                print(feature, range_)
                mask = (df_features[feature] >= range_[0]) & (df_features[feature] <= range_[1])
                addresses_in_category.append(df_features[mask].index)

            # get the intersection of the addresses
            addresses_in_category = set(addresses_in_category[0]).intersection(*addresses_in_category[1:])
            category_data[category] = addresses_in_category




        # create df, rows are the addresses, columns are the categories
        category_df = pd.DataFrame(index=df_features.index)
        for category, addresses in category_data.items():
            category_df[category] = 0
            category_df.loc[addresses, category] = 1



        return category_df, df_desc

    @log_time
    def get_clustering(self, prefix, configs, algorithm, nafill, preprocessing, n_cluster, df_normalized, df_eval_normalized):

        fp = FeaturesModeller(configs, prefix, nafill)

        if preprocessing == "UMAP":
            data, data_eval = self.embed(prefix, configs, df_normalized, df_eval_normalized, nafill, cache=True)
        else:
            data, data_eval = df_normalized, df_eval_normalized


        if algorithm == "gmm":
            fp.fit_GMM(data, n_cluster)
            evalset_clustered = fp.transform_GMM(data_eval)
        elif algorithm == "kmeans":
            fp.fit_KMEANS(data, n_cluster)
            evalset_clustered = fp.transform_KMEANS(data_eval)
        else:
            raise ValueError("algorithm not known")
        return evalset_clustered, n_cluster



    @log_time
    def get_clustering_elbow(self, prefix, configs, algorithm, nafill, preprocessing, df_normalized, df_eval_normalized):

        fp = FeaturesModeller(configs, prefix, nafill)

        if preprocessing == "UMAP":
            data, data_eval = self.embed(prefix, configs, df_normalized, df_eval_normalized, nafill, cache=True)
        else:
            data, data_eval = df_normalized, df_eval_normalized


        if algorithm == "gmm":
            fp.fit_transform_BIC_optimal_GMM(data)
            evalset_clustered = fp.transform_GMM(data_eval)
            n_cluster = fp.get_GMM_n_clusters()
        elif algorithm == "kmeans":
            fp.fit_transform_KMEANS(data)
            evalset_clustered = fp.transform_KMEANS(data_eval)
            n_cluster = fp.get_KMEANS_n_clusters()
        else:
            raise ValueError("algorithm not known")
        return evalset_clustered, n_cluster


    def get_supervised_pipelines(self):


        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        pipeline1 = make_pipeline(StandardScaler(), imputer, RandomForestClassifier(n_estimators=400, random_state=0))
        pipeline2 = make_pipeline(StandardScaler(), imputer, XGBClassifier())
        pipeline3 = make_pipeline(StandardScaler(), imputer, AdaBoostClassifier())
        return pipeline1, pipeline2, pipeline3

    @log_time
    def experiments_supervised(self):

        # set seed for numpy and sklearn
        np.random.seed(0)

        evalset_features = self.agg.load_evalset_features()
        evalset_features_processed = self.fp.preprocess(evalset_features)
        evalset = self.fp.load_eval_set()


        #df_eval_filled = evalset_features_processed.fillna(-1)
        df_eval_reindexed = evalset_features_processed.reindex(evalset.index)

        X = df_eval_reindexed.values
        y = evalset["Bot"].values
        # Create a pipeline with scaling, filling missing values, and random forest classifier
        pipeline1, pipeline2, pipeline3 = self.get_supervised_pipelines()

        classes = evalset["fine-grained reconciled"]
        scores_rf, detailed, _ = get_scores(pipeline1, X, y, classes)

        pipeline1.fit(X,y)
        columns = df_eval_reindexed.columns

        data = (X, y, pipeline1, columns)
        filename = "evalset_supervised_shap"
        section = "results"
        save_data_for_figure(data, filename, section, self.prefix)

        scores = {
            "RandomForest": scores_rf,
            "GradientBoosting": get_scores(pipeline2, X, y, classes)[0],
            "AdaBoost": get_scores(pipeline3, X, y, classes)[0],

        }

        metrics_df = pd.DataFrame(scores).T
        metrics_df.index.name = "Algorithm"
        filename =  "evalset_supervised_scores"
        section = "results"
        save_table(metrics_df, filename, section, self.prefix, index=True)

        filename = "evalset_supervised_detailed"
        section = "results"
        save_table(detailed, filename, section, self.prefix, index=True)


        ### handle balanced dataset of arbitrages
        ### we get arbs from the transactions MEV inspect detected
        types = ["arbitrages", "sandwiches", "liquidations"]
        type_dataset_names = ["Arbitrage", "Sandwich", "Liquidation"]
        data_MEV = {}
        results_MEV = {}
        results_MEV_detailed = {}

        for type_i in range(len(types)):

            type = types[type_i]
            type_dataset_name = type_dataset_names[type_i]
            df_selecttypes = self.agg.load_MEVinspect(type)

            df_eval_selecttypes = evalset[evalset["fine-grained reconciled"] == type_dataset_name]
            df_eval_selecttypes_features = df_eval_reindexed.loc[df_eval_selecttypes.index]
            df_eval_nonselecttypes = evalset[evalset["fine-grained reconciled"] != type_dataset_name]
            df_eval_nonselecttypes_features = df_eval_reindexed.loc[df_eval_nonselecttypes.index]


            # drop test blocks data from the mev inspect data
            df_selecttypes_no_eval = df_selecttypes.drop(df_eval_selecttypes_features.index, errors="ignore")

            # for the case where there are more rows in the nonselecttypes than in the selecttypes
            n_rows_diff = df_eval_nonselecttypes_features.shape[0] - df_selecttypes_no_eval.shape[0]
            if n_rows_diff > 0:
                target_size = df_selecttypes_no_eval.shape[0]
                subset_indices = df_eval_nonselecttypes_features.sample(n=target_size, random_state=0).index
                df_eval_nonselecttypes, df_eval_nonselecttypes_features = df_eval_nonselecttypes.loc[subset_indices], df_eval_nonselecttypes_features.loc[subset_indices]
                #df_eval_selecttypes_features_processed = self.fp.preprocess(df_selecttypes_no_eval.sample(n=n_rows_diff_maxed, random_state=0))

                df_eval_selecttypes_features_processed = self.fp.preprocess(df_selecttypes_no_eval.sample(n=target_size, random_state=0))
            else:
                target_size = df_eval_nonselecttypes_features.shape[0]
                df_eval_selecttypes_features_processed = self.fp.preprocess(df_selecttypes_no_eval.sample(n=target_size, random_state=0))




            # save it to build multiclass MEV dataset
            data_MEV[type_dataset_name] = df_eval_selecttypes_features_processed


            # now concat the dataset the first is class 0, the second is class 1
            X = np.concatenate([df_eval_nonselecttypes_features, df_eval_selecttypes_features_processed])
            y = np.concatenate([np.zeros(df_eval_nonselecttypes_features.shape[0]), np.ones(df_eval_selecttypes_features_processed.shape[0])])
            # y as int
            y = y.astype(int)

            classes = list(df_eval_nonselecttypes["fine-grained reconciled"]) + len(df_eval_selecttypes_features_processed)*[type_dataset_name]
            negative_classes = set(evalset["fine-grained reconciled"].unique()) - set([type_dataset_name])
            scores_rf, detailed, category_matrix = get_scores(pipeline1, X, y, classes, negative_classes=negative_classes)

            results_MEV[type_dataset_name] = scores_rf
            results_MEV_detailed[type_dataset_name] = detailed

            # fit a random forest and get the feature importances
            pipeline1.fit(X, y)
            feature_importances = pipeline1.named_steps["randomforestclassifier"].feature_importances_
            feature_importances = pd.Series(feature_importances, index=df_eval_reindexed.columns)
            feature_importances = feature_importances.sort_values(ascending=False)
            feature_importances = feature_importances[feature_importances > 0]
            feature_importances = feature_importances / feature_importances.sum()
            feature_importances = feature_importances.sort_values(ascending=False)
            feature_importances = feature_importances.to_frame()
            feature_importances.columns = ["Importance"]
            feature_importances.index.name = "Feature"
            print(feature_importances)


        ### build multiclass model
        mask = [x not in type_dataset_names for x in evalset["fine-grained reconciled"]]
        df_eval_selecttypes = evalset[mask]
        df_eval_selecttypes_features = df_eval_reindexed.loc[df_eval_selecttypes.index]


        # build dataset from data and drop duplicates
        df_all_types = pd.concat(data_MEV)
        df_all_types.reset_index(inplace=True)
        df_all_types.rename(columns={"level_0": "type"}, inplace=True)
        df_all_types.index = df_all_types["address"]
        df_all_types.drop(columns=["address"], inplace=True)
        dup_mask = df_all_types.index.duplicated(keep=False)


        print(df_all_types[dup_mask])
        df_all_types = df_all_types[~dup_mask]

        # cut back subsets to the same size of the smallest subset
        min_size = df_all_types.groupby("type").size().min()
        df_all_types = df_all_types.groupby("type").apply(lambda x: x.sample(n=min_size, random_state=0))
        df_all_types.drop(columns=["type"], inplace=True)
        df_all_types.reset_index(inplace=True)
        df_all_types.set_index("address", inplace=True)
        df_all_types.index.name = "index"

        non_MEV = df_eval_selecttypes_features.sample(n=min_size, random_state=0)
        # following only for diagnostic purposes
        non_MEV_finegrained = df_eval_selecttypes.loc[non_MEV.index]["fine-grained reconciled"]

        non_MEV["type"] = "non-MEV"
        df_all_types_and_eval = pd.concat([df_all_types, non_MEV])

        save_data_for_figure(df_all_types_and_eval, "feature_difference_multiclass_MEV", "results", prefix=self.prefix)

        # prepare X and y for classification
        X = df_all_types_and_eval.drop(columns=["type"])
        y = df_all_types_and_eval["type"]

        # fit a random forest and get the feature importances
        pipeline1.fit(X, y)
        feature_importances_multiclass = pipeline1.named_steps["randomforestclassifier"].feature_importances_
        feature_importances_multiclass = pd.Series(feature_importances_multiclass, index=X.columns)
        feature_importances_multiclass = feature_importances_multiclass.sort_values(ascending=False)
        feature_importances_multiclass = feature_importances_multiclass[feature_importances_multiclass > 0]
        feature_importances_multiclass = feature_importances_multiclass / feature_importances_multiclass.sum()
        feature_importances_multiclass = feature_importances_multiclass.sort_values(ascending=False)
        feature_importances_multiclass = feature_importances_multiclass.to_frame()
        feature_importances_multiclass.columns = ["Importance"]
        feature_importances_multiclass.index.name = "Feature"
        print(feature_importances_multiclass)

        # accuracy, precision, recall, f1, roc_auc
        # map y as int
        classes_unique = y.unique()
        y_num = y.map({i: idx for idx, i in enumerate(classes_unique)})
        scores_rf, detailed, category_matrix = get_scores(pipeline1, X.values, y.values, y.values, multiclass=True)
        scores_gb, detailed_gb, category_matrix_gb = get_scores(pipeline2, X.values, y.values, y.values, multiclass=True)

        scores_multiclass = {
            "RandomForest": scores_rf,
            "GradientBoosting": scores_gb,
            "AdaBoost": get_scores(pipeline3, X.values, y.values, y.values, multiclass=True)[0],

        }


        results_MEV_df = pd.DataFrame(results_MEV)
        results_MEV_df = results_MEV_df.T
        results_MEV_df.index.name = "MEV-type"
        results_MEV_df.columns.name = ""
        metrics_df.index.name = "Algorithm"

        results_MEV_detailed_df = pd.concat(results_MEV_detailed).T
        results_MEV_detailed_df.fillna(0, inplace=True)
        for type_name in type_dataset_names:
            results_MEV_detailed_df[type_name, "N Samples"] =  results_MEV_detailed_df[type_name, "N Samples"].astype(int)

        results_MEV_detailed_df.index.name = "EOA-Type"
        results_MEV_detailed_df.columns.names = ["", ""]

        filename = "MEV_metrics"
        section = "results"
        save_table(results_MEV_df, filename, section, self.prefix, index=True)

        filename =  "MEV_metrics_detailed"
        section = "results"
        save_table(results_MEV_detailed_df, filename, section, self.prefix, index=True)


        df_metrics_multiclass = pd.DataFrame(scores_multiclass).T
        #df_metrics_multiclass.columns.name = ""
        df_metrics_multiclass.index.name = "Algorithm"
        filename =  "MEV_metrics_multiclass"
        section = "results"
        save_table(df_metrics_multiclass, filename, section, self.prefix, index=True)

        # plot confusion matrix
        #convert to percentage too
        category_matrix_normalised = category_matrix / category_matrix.sum(axis=1)
        filename = "confusion_matrix_MEV_multiclass"
        chapter = "results"
        save_data_for_figure(category_matrix_normalised, filename, chapter, prefix=self.prefix)

        # do the same for gradient boosting
        category_matrix_normalised = category_matrix_gb / category_matrix_gb.sum(axis=1)
        filename = "confusion_matrix_MEV_multiclass_gb"
        chapter = "results"
        save_data_for_figure(category_matrix_normalised, filename, chapter, prefix=self.prefix)

    @log_time
    def create_cluster_dfs(self):

        run = self.configs["General"]["run_name"]
        algorithms = self.configs["Clustering"]["algorithms"]
        nafills = self.configs["Clustering"]["nafills"]
        preprocessings = self.configs["Clustering"]["preprocessings"]
        n_clusters = [5, 15, 30]

        agg = Aggregate(self.configs, prefix=self.prefix)
        fp = FeaturesModeller(self.configs, prefix=self.prefix, nafill_type=self.configs["Heuristics"]["nafill"])

        df_features, df_normalized, df_eval_normalized, evalset = self.load_data(agg, fp)

        folder = f"{self.prefix}/outputs/{run}/cluster_results"
        if not os.path.exists(folder):
            os.makedirs(folder)

        # get all combinations
        combinations = list(itertools.product(algorithms, nafills, preprocessings, n_clusters))

        #df_normalized = df_normalized.sample(n=10000, random_state=0)

        for algorithm, nafill, preprocessing, n_cluster in combinations:
            # make reproducible
            np.random.seed(0)
            print(algorithm, nafill, preprocessing, n_cluster)
            evalset_clustered, n_clusters = self.get_clustering(self.prefix, self.configs, algorithm, nafill, preprocessing, n_cluster, df_normalized, df_eval_normalized)
            evalset_clustered.to_csv(f"{self.prefix}/outputs/{run}/cluster_results/evalset_clustered_{algorithm}_{nafill}_{preprocessing}_{n_cluster}.csv")
            # get the number of clusters and write it to a file
            path = f"{self.prefix}/outputs/{run}/cluster_results/evalset_clustered_nclusters_{algorithm}_{nafill}_{preprocessing}_{n_cluster}.csv"
            with open(path, "w") as f:
                f.write(str(n_clusters))

    @log_time
    def create_cluster_dfs_elbow(self):

        run = self.configs["General"]["run_name"]
        algorithms = self.configs["Clustering"]["algorithms"]
        nafills = self.configs["Clustering"]["nafills"]
        preprocessings = self.configs["Clustering"]["preprocessings"]

        agg = Aggregate(self.configs, prefix=self.prefix)
        fp = FeaturesModeller(self.configs, prefix=self.prefix, nafill_type=self.configs["Heuristics"]["nafill"])

        df_features, df_normalized, df_eval_normalized, evalset = self.load_data(agg, fp)

        folder = f"{self.prefix}/outputs/{run}/cluster_results"
        if not os.path.exists(folder):
            os.makedirs(folder)

        # get all combinations
        combinations = list(itertools.product(algorithms, nafills, preprocessings))

        #df_normalized = df_normalized.sample(n=1000, random_state=0)

        for algorithm, nafill, preprocessing in combinations:
            # make reproducible
            np.random.seed(0)
            print("elbow", algorithm, nafill, preprocessing)
            evalset_clustered, n_clusters = self.get_clustering_elbow(self.prefix, self.configs, algorithm, nafill,
                                                                preprocessing, df_normalized,
                                                                df_eval_normalized)
            evalset_clustered.to_csv(
                f"{self.prefix}/outputs/{run}/cluster_results/evalset_elbow_clustered_{algorithm}_{nafill}_{preprocessing}.csv")
            # get the number of clusters and write it to a file
            path = f"{self.prefix}/outputs/{run}/cluster_results/evalset_elbow_clustered_nclusters_{algorithm}_{nafill}_{preprocessing}.csv"
            with open(path, "w") as f:
                f.write(str(n_clusters))

    def create_cluster_table_elbow(self):
        """
        Load the cluster results and create a table with the metrics and the number of clusters
        :return:
        """

        run = self.configs["General"]["run_name"]
        algorithms = self.configs["Clustering"]["algorithms"]
        nafills = self.configs["Clustering"]["nafills"]
        preprocessings = self.configs["Clustering"]["preprocessings"]
        # get all combinations
        combinations = list(itertools.product(algorithms, nafills, preprocessings))
        folder = f"{self.prefix}/outputs/{run}/cluster_results"
        if not os.path.exists(folder):
            os.makedirs(folder)

        evalset = self.fp.load_eval_set()

        data = {}
        for algorithm, nafill, preprocessing in combinations:
            df_raw_clusters = pd.read_csv(f"{self.prefix}/outputs/{run}/cluster_results/evalset_elbow_clustered_{algorithm}_{nafill}_{preprocessing}.csv", index_col=0)


            results = self.evaluate_clustering_from_dfs_light(evalset, df_raw_clusters)

            data[(algorithm, nafill, preprocessing)] = results



        df = pd.DataFrame(data).T
        filename = "clustering_results_purity_elbow"
        section = "results"
        save_table(df, filename, section, self.prefix, index=True)



    def create_cluster_table_clustersizefixed(self):
        """
        Load the cluster results and create a table with the metrics and the number of clusters
        :return:
        """

        run = self.configs["General"]["run_name"]
        algorithms = self.configs["Clustering"]["algorithms"]
        nafills = self.configs["Clustering"]["nafills"]
        preprocessings = self.configs["Clustering"]["preprocessings"]
        n_clusters = [5, 15, 30]

        # get all combinations
        combinations = list(itertools.product(algorithms, nafills, preprocessings, n_clusters))
        folder = f"{self.prefix}/outputs/{run}/cluster_results"
        if not os.path.exists(folder):
            os.makedirs(folder)

        evalset = self.fp.load_eval_set()

        data = {}

        data_rf = {}
        for algorithm, nafill, preprocessing, n_cluster in combinations:
            df_raw_clusters = pd.read_csv(f"{self.prefix}/outputs/{run}/cluster_results/evalset_clustered_{algorithm}_{nafill}_{preprocessing}_{n_cluster}.csv", index_col=0)

            # order df_raw_clusters by the index of evalset
            df_raw_clusters = df_raw_clusters.reindex(evalset.index)
            #X = df_raw_clusters["cluster"].values.reshape(-1,1)
            #y = evalset["Bot"].values

            #clf = RandomForestClassifier(n_estimators=400, random_state=0)

            # Generate cross-validated predictions
            #cv = 20

            #scoring = {
            #    'accuracy': make_scorer(accuracy_score),
            #    'precision': make_scorer(precision_score),#, average="macro"),
            #    'recall': make_scorer(recall_score),#, average="macro"),
            #    'f1': make_scorer(f1_score),#, average="macro")
            #}

            #scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)

            #accuracy_cv_samples = scores['test_accuracy']
            #precision_cv_samples = scores['test_precision']
            #recall_cv_samples = scores['test_recall']
            #f1_cv_samples = scores['test_f1']

            #accuracy_str = reportstring(accuracy_cv_samples)
            #precision_str = reportstring(precision_cv_samples)
            #recall_str = reportstring(recall_cv_samples)
            #f1_str = reportstring(f1_cv_samples)

            #data_rf[(algorithm, nafill, preprocessing)] = {"Accuracy": accuracy_str, "Precision": precision_str, "Recall": recall_str, "F1": f1_str}

            results = self.evaluate_clustering_from_dfs_light(evalset, df_raw_clusters)

            data[(algorithm, nafill, preprocessing, n_cluster)] = results



        df = pd.DataFrame(data).T
        # transform the last level to additional columns
        print(df)
        filename = "clustering_results_purity_clustersizefixed"
        section = "results"
        save_table(df, filename, section, self.prefix, index=True)

        df_rf = pd.DataFrame(data_rf).T
        print(df_rf)
        filename = "clustering_results_rf_clustersizefixed"
        section = "results"
        save_table(df_rf, filename, section, self.prefix, index=True)

        #### get the combination with the highest accuracy (purity)
        max_accuracy = 0
        best_combination = None
        for combination, results in data.items():
            if results["Purity"] > max_accuracy:
                max_accuracy = results["Purity"]
                best_combination = combination

        self.logger.info(f"Best combination: {best_combination} with accuracy {max_accuracy}")

        # get detailed results for the best combination
        print(best_combination)
        algorithm, nafill, preprocessing, n_cluster = best_combination
        df_raw_clusters = pd.read_csv(f"{self.prefix}/outputs/{run}/cluster_results/evalset_clustered_{algorithm}_{nafill}_{preprocessing}_{n_cluster}.csv", index_col=0)
        results, assigned = self.evaluate_clustering_from_dfs(evalset, df_raw_clusters)

        assigned.rename(columns={"label": "predicted", "cluster_id": "category"}, inplace=True)

        category_matrix, bot_or_not_matrix, detailed_eval, accuracy, df_purity_entropy = self.classification_details(assigned, evalset)

        # save data for the best combination
        filename = "clustering_results_best_combination_category_matrix_clustersizefixed"
        section = "results"
        save_table(category_matrix, filename, section, self.prefix, index=True)

        filename = "clustering_results_best_combination_purity_entropy_clustersizefixed"
        section = "results"
        save_table(df_purity_entropy, filename, section, self.prefix, index=True)

        return best_combination



    def classification_details(self, evalset_predicted, evalset):

        # rename the clusters, saved in the "category" column ascending from 0 to n, where n is the number of different clusters assigned for the test set only

        # get the unique cluster ids
        unique_clusters = evalset_predicted["category"].unique()
        # sort them
        unique_clusters.sort()
        # create a mapping from the cluster id to a new id, starting from 0
        mapping = {cluster: i for i, cluster in enumerate(unique_clusters)}
        # apply the mapping to the category column
        evalset_predicted["category"] = evalset_predicted["category"].map(mapping)



        evalset_predicted_fine = pd.merge(evalset, evalset_predicted, left_index=True, right_index=True, how="left")

        category_matrix = pd.crosstab(evalset_predicted_fine["category"],
                                      evalset_predicted_fine["fine-grained reconciled"])

        accuracy = accuracy_prediction(evalset_predicted_fine["predicted"], evalset["Bot"])

        evalset_predicted["correct"] = evalset_predicted["predicted"] == evalset["Bot"]
        evalset_predicted["wrong"] = ~evalset_predicted["correct"]
        evalset_predicted["finegrained"] = evalset_predicted_fine["fine-grained reconciled"]

        # Calculate the "finegrained" metrics for correct and wrong classifications
        detailed_eval = evalset_predicted.groupby("finegrained")[["correct", "wrong"]].mean()


        bot_or_not_matrix = pd.crosstab(evalset_predicted_fine["predicted"],
                                      evalset_predicted_fine["fine-grained reconciled"])



        purity_scores = purity_scores_single(evalset["Bot"], evalset_predicted_fine["category"])
        entropy_scores = entropy_scores_single(evalset["Bot"], evalset_predicted_fine["category"])

        df_purity_entropy = pd.DataFrame({"purity": purity_scores, "entropy": entropy_scores}, index=range(len(purity_scores)))
        df_purity_entropy.index.name = "cluster"
        numbers = category_matrix.sum(axis=1)
        df_purity_entropy["size"] = numbers



        return category_matrix, bot_or_not_matrix, detailed_eval, accuracy, df_purity_entropy

    def threshold_based_recall_precision(self, df_features_evalset, target):
        df_features_evalset = df_features_evalset.copy()

        # e.g. super low trade value clustering only makes sense if there are sufficiently many trades
        features = [
            "tx__value__tvc"
            , "tx__custom__n_tx_per_block"
            , "scb__eb__transfer__value__tvc"
            , "tx__time__intime_sleepiness"
            , "tx__time__outtime_sleepiness"
            , "tx__time__intime_transaction_frequency"
            , "tx__time__outtime_transaction_frequency"

        ]
        #df_features_evalset = df_features_evalset[features].dropna()
        high_is_suspicious_list = [
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            True
        ]

        dfs = []
        # for one feature, go through cut-off values and create roc curve
        for feature, high_is_suspicious in zip(features, high_is_suspicious_list):
            # create curve
            data = []
            # order
            features_vec = df_features_evalset[feature]
            # order target the same way

            if high_is_suspicious:
                # fill up na with min
                features_vec_ordered = features_vec.sort_values(ascending=False)
                features_vec_ordered.fillna(features_vec_ordered.min(), inplace=True)
            else:
                # fill up na with max
                features_vec_ordered = features_vec.sort_values(ascending=True)
                features_vec_ordered.fillna(features_vec_ordered.max(), inplace=True)

            target = target.loc[features_vec_ordered.index]

            for i in range(len(features_vec_ordered)):


                if high_is_suspicious:
                    mask = features_vec_ordered >= features_vec_ordered.iloc[i]
                else:
                    mask = features_vec_ordered <= features_vec_ordered.iloc[i]

                preds = np.zeros(len(df_features_evalset))
                preds[mask] = 1
                res = classification_metrics_from_binary_predictions(target, preds)
                data.append(res)

            df = pd.DataFrame(data)
            dfs.append(df)
            #df.drop_duplicates(inplace=True)
            #plt.plot(df["Recall"], df["Precision"], label=feature)

        #plt.legend()
        #plt.show()

        data = dfs, features
        filename = "threshold_based_recall_precision"
        section = "results"
        save_data_for_figure(data, filename, section, self.prefix)

    """
        for feature, high_is_suspicious in zip(features, high_is_suspicious_list):
            # create curve
            data = []
            # order
            mask = df_features_evalset[feature].isna()
            features_vec = df_features_evalset[feature]
            features_vec = features_vec[~mask]
            # order target the same way

            if high_is_suspicious:
                # fill up na with min
                features_vec_ordered = features_vec.sort_values(ascending=False)
            else:
                # fill up na with max
                features_vec_ordered = features_vec.sort_values(ascending=True)

            target_ = target.loc[features_vec_ordered.index]

            for i in range(len(features_vec_ordered)):


                if high_is_suspicious:
                    mask = features_vec_ordered >= features_vec_ordered.iloc[i]
                else:
                    mask = features_vec_ordered <= features_vec_ordered.iloc[i]

                preds = np.zeros(len(features_vec_ordered))
                preds[mask] = 1
                res = classification_metrics_from_binary_predictions(target_, preds)
                data.append(res)
                
            df = pd.DataFrame(data)
            dfs.append(df)
            df.drop_duplicates(inplace=True)
            plt.plot(df["Recall"], df["Precision"], label=feature)
        """

    def threshold_based_recall_precision_paper(self, df_features_evalset, target):
        df_features_evalset = df_features_evalset.copy()

        # e.g. super low trade value clustering only makes sense if there are sufficiently many trades
        features = [
            "tx__time__outtime_hourly_entropy"
            , "tx__custom__n_tx_per_block"
            , "tx__time__outtime_transaction_frequency"
            , "tx__generic__gas_price_max"
            , "tx__time__outtime_sleepiness"
        ]


        #df_features_evalset = df_features_evalset[features].dropna()
        high_is_suspicious_list = [
            True,
            True,
            True,
            True,
            False
        ]

        dfs = []
        # for one feature, go through cut-off values and create roc curve
        for feature, high_is_suspicious in zip(features, high_is_suspicious_list):
            # create curve
            data = []
            # order
            features_vec = df_features_evalset[feature]
            # order target the same way

            if high_is_suspicious:
                # fill up na with min
                features_vec_ordered = features_vec.sort_values(ascending=False)
                features_vec_ordered.fillna(features_vec_ordered.min(), inplace=True)
            else:
                # fill up na with max
                features_vec_ordered = features_vec.sort_values(ascending=True)
                features_vec_ordered.fillna(features_vec_ordered.max(), inplace=True)

            target = target.loc[features_vec_ordered.index]

            for i in range(len(features_vec_ordered)):


                if high_is_suspicious:
                    mask = features_vec_ordered >= features_vec_ordered.iloc[i]
                else:
                    mask = features_vec_ordered <= features_vec_ordered.iloc[i]

                preds = np.zeros(len(df_features_evalset))
                preds[mask] = 1
                res = classification_metrics_from_binary_predictions(target, preds)
                data.append(res)

            df = pd.DataFrame(data)
            dfs.append(df)
            #df.drop_duplicates(inplace=True)
            #plt.plot(df["Recall"], df["Precision"], label=feature)

        #plt.legend()
        #plt.show()

        data = dfs, features
        filename = "threshold_based_recall_precision_paper"
        section = "results"
        save_data_for_figure(data, filename, section, self.prefix)



    @log_time
    def experiments_threshold_based(self):

        # get data
        evalset = self.fp.load_eval_set()
        df_features_evalset = self.agg.load_evalset_features()
        df_features_evalset = df_features_evalset.loc[evalset.index]

        # get the threshold based results
        self.fp.preprocess(df_features_evalset)
        threshold_based_results, df_threshold_desc = self.threshold_based(df_features_evalset)

        target = evalset["Bot"]
        self.threshold_based_recall_precision_paper(df_features_evalset, target)
        self.threshold_based_recall_precision(df_features_evalset, target)


        threshold_based_results["Combined"] = threshold_based_results.max(axis=1)
        threshold_based_results["fine-grained reconciled"] = evalset["fine-grained reconciled"]
        df_threshold_results_detailed = threshold_based_results.groupby("fine-grained reconciled").sum() / threshold_based_results.groupby("fine-grained reconciled").count()
        df_threshold_results_detailed["N Samples"] = threshold_based_results.groupby("fine-grained reconciled").count()["Combined"].astype(int)
        data = []
        y_test = evalset["Bot"]
        threshold_based_names = threshold_based_results.columns[:-1]
        for col in threshold_based_names:
            y_pred = threshold_based_results[col]
            data += [
                classification_metrics_from_binary_predictions(y_test, y_pred)
            ]

        df_threshold_results = pd.DataFrame(data, index=threshold_based_names)
        # note that precision is much more valuable in such a system than recall, because if the precision is high it
        # can be used as an add-on to an existing system, without costing false positives.
        filename = "heuristics_threshold_based_results"
        section = "results"
        save_table(df_threshold_results, filename, section, self.prefix, index=True)

        filename = "heuristics_threshold_based_results_detailed"
        section = "results"
        # map the column names to ascending ints

        save_table(df_threshold_results_detailed, filename, section, self.prefix, index=True)

        filename = "heuristics_threshold_based_description"
        section = "results"
        # replace & with & \n
        df_threshold_desc = df_threshold_desc.replace("&", "& \\n")

        with pd.option_context("max_colwidth", 1000):

           save_table(df_threshold_desc, filename, section, self.prefix, index=False)


    @log_time
    def experiments_supervised_with_cluster_info(self, top_setup):

        np.random.seed(42)
        run = self.configs["General"]["run_name"]
        folder = f"{self.prefix}/outputs/{run}/cluster_results"
        if not os.path.exists(folder):
            os.makedirs(folder)

        algorithm, nafill, preprocessing = top_setup
        df_raw_clusters = pd.read_csv(f"{self.prefix}/outputs/{run}/cluster_results/evalset_clustered_{algorithm}_{nafill}_{preprocessing}.csv", index_col=0)


        evalset_features = self.agg.load_evalset_features()
        evalset_features_processed = self.fp.preprocess(evalset_features)
        evalset = self.fp.load_eval_set()

        df_eval_reindexed = evalset_features_processed.reindex(evalset.index)
        df_raw_clusters_reindexed = df_raw_clusters.reindex(evalset.index)

        new_cols = pd.get_dummies(df_raw_clusters_reindexed, columns=["cluster"])
        df_eval_reindexed = pd.concat([df_eval_reindexed, new_cols], axis=1)

        X = df_eval_reindexed.values

        y = evalset["Bot"].values
        # Create a pipeline with scaling, filling missing values, and random forest classifier
        pipeline1, pipeline2,pipeline3 = self.get_supervised_pipelines()

        classes = evalset["fine-grained reconciled"]
        scores_rf, detailed, _ = get_scores(pipeline1, X, y, classes)


        scores = {
            "RandomForest": scores_rf,
            "GradientBoosting": get_scores(pipeline2, X, y, classes)[0],
            "AdaBoost": get_scores(pipeline3, X, y, classes)[0],

        }

        metrics_df = pd.DataFrame(scores).T
        metrics_df.index.name = "Algorithm"
        filename =  "evalset_with_clusterinfo_supervised_scores"
        section = "results"
        save_table(metrics_df, filename, section, self.prefix, index=True)


    @log_time
    def experiments_graph_based_calculation(self):

        configs = load_configs(self.prefix)
        df_features, df_normalized, df_eval_normalized, evalset = self.load_data(self.agg, self.fp)
        categories_drs = self.graph_based_deposit_receiver_supplier(df_features, configs)

        save_data(categories_drs, "categories_deposit_receiver_supplier", self.prefix)
        categories_diamond = self.graph_based_diamond_pattern()
        save_data(categories_diamond, "categories_diamond", self.prefix)


    @log_time
    def experiments_graph_based_evaluate(self):

        evalset = self.fp.load_eval_set()
        categories_drs = load_data("categories_deposit_receiver_supplier", self.prefix)
        categories_diamond = load_data("categories_diamond", self.prefix)


        print(categories_drs.reset_index().groupby("category").count())
        print(categories_diamond.reset_index().groupby("category").count())

        unified_categories = self.unify_categories([categories_drs, categories_diamond])

        def postprocess_merge(df):
            df = df.copy()
            mask = df["category"].notna()
            df["category"][mask] = 1
            df["category"].fillna(0, inplace=True)
            df["category"] = df["category"].astype(int)
            df.rename(columns={"category": "predicted"}, inplace=True)
            df = df[["predicted"]]
            return df

        #def metrics(df):
        #    precision = precision_score(evalset["Bot"], df, average="macro")
        #    recall = recall_score(evalset["Bot"], df, average="macro")
        #    f1 = f1_score(evalset["Bot"], df, average="macro")
        #    accuracy = accuracy_score(evalset["Bot"], df)
        #    data = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}
        #    return data

        def merge_eval(df):

            evalset_predicted_fine = pd.merge(evalset, df, left_index=True, right_index=True, how="left")
            evalset_predicted = postprocess_merge(evalset_predicted_fine)
            return classification_metrics_from_binary_predictions(evalset["Bot"].values, evalset_predicted.values.reshape(-1))# metrics(evalset_predicted)

        data = [merge_eval(categories_drs), merge_eval(categories_diamond), merge_eval(unified_categories)]
        # set "middle" in diamond to zero
        categories_diamond["category"][categories_diamond["category"] == "middle"] = np.nan
        categories_drs["category"][categories_drs["category"] == "deposit_wallet"] = np.nan
        unified_categories = self.unify_categories([categories_drs, categories_diamond])
        data += [merge_eval(categories_drs), merge_eval(categories_diamond), merge_eval(unified_categories)]

        df = pd.DataFrame(data, index=["CEX-Pattern", "Diamond", "Both", "CEX-Pattern-small", "Diamond-small", "Both-small"])
        print(df)
        df.index.name = "Algorithm"
        filename = "evalset_graph_based_scores"
        section = "results"
        save_table(df, filename, section, self.prefix, index=True)

    @log_time
    def run_experiments(self):

        ##########################
        ### THRESHOLD BASED
        self.experiments_threshold_based()

        ##########################
        ### GRAPH BASED
        self.experiments_graph_based_calculation()
        self.experiments_graph_based_evaluate()

        ##########################
        ### SUPERVISED
        self.experiments_supervised()

        ##########################
        ### CLUSTERING
        self.create_cluster_dfs()
        top_setup = self.create_cluster_table()

        # clustering and supervised results

        self.experiments_supervised_with_cluster_info(top_setup)


    @log_time
    def train(self):

        np.random.seed(0)
        ### FIT RANDOM FOREST
        evalset_features = self.agg.load_evalset_features()
        # get empty df that just contains columns
        col_df = pd.DataFrame(columns=[evalset_features.index.name] + list(evalset_features.columns))
        # add address in the first
        with open(self.prefix + "/models/col_names.pkl", "wb") as f:
            pickle.dump(col_df, f)

        preprocessing_function = self.fp.preprocess
        mainwindow_features_processed = self.agg.load_features(preprocessing_function=preprocessing_function) #  for fitting a scaler on the larger window

        evalset_features_processed = self.fp.preprocess(evalset_features)
        evalset = self.fp.load_eval_set()

        df_eval_reindexed = evalset_features_processed.reindex(evalset.index)
        # fix columns
        mainwindow_features_processed_processed = mainwindow_features_processed.reindex(df_eval_reindexed.columns, axis=1)


        #X = df_eval_reindexed.values
        y = evalset["Bot"].values
        # Fill NAs
        pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=400, random_state=0))
        scaler = pipeline.steps[0][1]
        model = pipeline.steps[1][1]

        # fit the scaler on the larger dataset to mitigate bias for the small test blocks set which has skewed distributions
        scaler.fit(mainwindow_features_processed_processed)
        X_transformed = scaler.transform(df_eval_reindexed)
        X_filled = np.where(np.isnan(X_transformed), 0, X_transformed)

        model.fit(X_filled, y)

        # pickle the pipeline
        with open(self.prefix + "/models/randomforest_pipeline.pkl", "wb") as f:
            pickle.dump(pipeline, f)



    @log_time
    def load_models(self):

        ### LOAD RANDOM FOREST pipeline
        with open(self.prefix + "/models/randomforest_pipeline.pkl", "rb") as f:
            randomforest_pipeline = pickle.load(f)

        return randomforest_pipeline

    @log_time
    def evaluate(self):
        randomforest_pipeline = self.load_models()

        with open(self.prefix + "/models/col_names.pkl", "rb") as f:
            col_df = pickle.load(f)

        # load features of entire dataset
        # partially fill in the second parameter of the function


        preprocessing_function = partial(self.fp.preprocess, desired_cols=col_df.columns)
        df_features = self.agg.load_features(preprocessing_function=preprocessing_function)

        scaler = randomforest_pipeline.steps[0][1]
        reduced_columns = scaler.feature_names_in_
        randomforest = randomforest_pipeline.steps[1][1]


        #df_features = self.fp.preprocess(df_features, col_df.columns)

        #X = df_features.values
        df_features_X = df_features.copy()
        # reorder the columns to match the order of the columns in the training set
        df_features_X = df_features_X[reduced_columns]

        #X_filled = np.where(np.isnan(X), 0, X)


        #### 1 - use the pipeline with the scaler
        X_scaled = scaler.transform(df_features_X)
        X_filled = np.where(np.isnan(X_scaled), 0, X_scaled)
        X_clipped = np.clip(X_filled, np.finfo(np.float32).min, np.finfo(np.float32).max)
        y_hat_randomforest_keep_scaling = randomforest.predict(X_clipped)

        # save the results
        df_results_randomforest_keep_scaling = pd.DataFrame(y_hat_randomforest_keep_scaling, index=df_features.index, columns=["bot"])
        filename = "randomforest_results_keep_scaling"
        save_prediction_results(df_results_randomforest_keep_scaling, filename, self.prefix)
        self.logger.info(f"main window scaling predicted bots: {df_results_randomforest_keep_scaling.sum()}, number of predicted non-bots: {len(df_results_randomforest_keep_scaling) - df_results_randomforest_keep_scaling.sum()}")

        #### 2 - use the pipeline without the scaler, and fit the scaler on the new data
        X_scaled = StandardScaler().fit_transform(df_features_X)
        X_filled = np.where(np.isnan(X_scaled), 0, X_scaled)
        X_clipped = np.clip(X_filled, np.finfo(np.float32).min, np.finfo(np.float32).max)

        y_hat_randomforest_new_scaling = randomforest.predict(X_clipped)

        # save the results
        df_results_randomforest_new_scaling = pd.DataFrame(y_hat_randomforest_new_scaling, index=df_features.index, columns=["bot"])
        filename = "randomforest_results_new_scaling"
        save_prediction_results(df_results_randomforest_new_scaling, filename, self.prefix)
        self.logger.info(f"window specific scaling predicted bots: {df_results_randomforest_new_scaling.sum()}, number of predicted non-bots: {len(df_results_randomforest_new_scaling) - df_results_randomforest_new_scaling.sum()}")

    def calculate_feature_stats(self) -> None:

        # load features of entire dataset
        df_features = self.agg.load_features(preprocessing_function = self.fp.preprocess)
        feature_means = pd.DataFrame(df_features.mean(axis=0, skipna=True))
        feature_stds = pd.DataFrame(df_features.std(axis=0, skipna=True))
        save_data(feature_means, "feature_means", self.prefix)
        save_data(feature_stds, "feature_stds", self.prefix)

    def get_accs(self) -> None:
        # get number of EOAs and CAs
        query = f"""
            SELECT address, account_type FROM accounts
        """
        self.connect_databases()
        self.cur.execute(query)
        results = self.cur.fetchall()
        self.disconnect_databases()
        results = pd.DataFrame(results, columns=["address", "account_type"])
        return results

    def calculate_acc_statistics(self, dfs: list) -> None:

        df = pd.concat(dfs)
        # deduplicate
        df = df.drop_duplicates(subset=["address"])
        # get number of EOAs and CAs
        counts = df.groupby("account_type").count()
        counts = counts["address"]
        counts = pd.DataFrame(counts)
        counts = counts.T
        counts.index.name = "count"
        save_data(counts, "acc_statistics", self.prefix)

    def calculate_DB_statistics(self) -> None:

        query = f"""
        SELECT COUNT(*) FROM "transferErc20"
        """
        self.connect_databases()
        self.cur.execute(query)
        tokencount = self.cur.fetchone()[0]
        self.disconnect_databases()

        query = f"""
            SELECT COUNT(*) FROM "transferEth"
            where value > 0
        """
        self.connect_databases()
        self.cur.execute(query)
        ethtransfercount = self.cur.fetchone()[0]
        self.disconnect_databases()

        # transactions overall
        query = f"""
            SELECT COUNT(*) FROM transactions
        """
        self.connect_databases()
        self.cur.execute(query)
        txcount = self.cur.fetchone()[0]
        self.disconnect_databases()

        # get number of EOAs and CAs
        query = f"""
            SELECT account_type, COUNT(*) FROM accounts
            GROUP BY account_type
        """
        self.connect_databases()
        self.cur.execute(query)
        results = self.cur.fetchall()
        self.disconnect_databases()
        results = pd.DataFrame(results, columns=["account_type", "count"])
        results.set_index("account_type", inplace=True)
        EOA_count = results.loc["EOA"]["count"]
        CA_count = results.loc["CA"]["count"]

        counts = {
            "EOA": EOA_count,
            "CA": CA_count,
            "TX": txcount,
            "ETH Transfer": ethtransfercount,
            "Token Transfer": tokencount
        }
        df_counts = pd.DataFrame(counts, index=[0])
        df_counts.index.name = "count"

        save_data(df_counts, "DB_statistics", self.prefix)

    def load_DB_statistics(self):
        DB_statistics = load_data("DB_statistics", self.prefix)
        return DB_statistics

    def load_acc_statistics(self):
        return load_data("acc_statistics", self.prefix)

    def load_feature_stats(self):
        feature_means = load_data("feature_means", self.prefix)
        feature_stds = load_data("feature_stds", self.prefix)
        return feature_means, feature_stds


    @log_time
    def get_transactions_with_botstatus_and_prices_SQL(self, prediction_file, cache=True):

        cache_filename = f'{prediction_file}_SQL_prices_botstatus.pkl.pbz2'


        def compress_pickle(title, data):
            with bz2.BZ2File(title, 'w') as f:
                cPickle.dump(data, f)

        def decompress_pickle(title):
            data = bz2.BZ2File(title, 'rb')
            data = cPickle.load(data)
            return data

        if cache:
            cache_folder = self.configs["General"]["PREFIX_DB"] + "/cache"
            if not os.path.exists(cache_folder):
                os.makedirs(cache_folder)

            cache_path = f'{cache_folder}/{cache_filename}'
            if os.path.exists(cache_path):
                results = decompress_pickle(cache_path)
                df_joined, df_joined_tokens, df_tokens, df_prices = results
                return df_joined, df_joined_tokens, df_tokens, df_prices


       # load prediction result
        df_results = load_prediction_results(prediction_file, self.prefix)

        ### sql create table from the df results, then join it with the table "
        # so the table will be text, boolean
        tablename = "prediction_results"
        delete = f"DROP TABLE IF EXISTS {tablename};"
        create = f"""
        CREATE TABLE IF NOT EXISTS {tablename} (
        address TEXT PRIMARY KEY,
        bot BOOLEAN
        );"""

        join =  f"""
            SELECT p.*, t.*, a.account_type as account_type_to_address
            FROM {tablename} AS p
            JOIN transactions AS t
            ON p.address = t.from_address
            LEFT JOIN accounts AS a
            ON t.to_address = a.address
        """

        create_index = f"""
        CREATE INDEX ON {tablename} (address);
        """

        load_prices = f'''
        SELECT * FROM  "priceData" 

        '''

        self.connect_databases()
        self.cur.execute(delete+create)
        self.conn.commit()

        start = time()
        df_results.to_sql(tablename, self.postgres_engine, method=psql_insert_df, if_exists="append", index=False)
        print(f"inserted in {time() - start} seconds")

        start = time()
        self.cur.execute(create_index)
        self.conn.commit()
        print(f"created index in {time() - start} seconds")
        start = time()

        df_joined = pd.read_sql(join, self.conn)
        print(f"joined in {time() - start} seconds")


        start = time()
        df_prices = pd.read_sql(load_prices, self.conn)
        print(f"loaded prices in {time() - start} seconds")

        path_to_tokens = f"{self.prefix}/data_lightweight/token_addresses.csv"
        df_tokens = pd.read_csv(path_to_tokens)

        query_tokens = f'''
             SELECT * FROM logs
             JOIN "transferErc20"   ON logs.block_id = "transferErc20".block_id
                                    AND logs.log_index = "transferErc20".log_index



             LEFT JOIN transactions AS t ON logs.transaction_index = t.transaction_index AND t.block_id = logs.block_id
             JOIN {tablename} AS p ON p.address = "transferErc20".from
             WHERE logs.address in {tuple([x.lower() for x in df_tokens.address])}
             AND t.receipt_status = 1
         '''

        df_joined_tokens = pd.read_sql(query_tokens, self.conn)


        # BE CAREFUL WITH THIS WHEN CHANGING THE ABOVE QUERY:
        # BUT: THERE IS NO SHORT SOLUTION TO THIS. best would be to add prefixes to the column names in the sql query when joining
        # but this would need more work
        # workaround
        # remove the second "value" column because that is for ETH, not tokens
        """ not necessary because its covered in the duplicate removal below
        mask = df_joined_tokens.columns == "value"
        # switch first True to False to keep the first column
        first_true_loc = np.where(mask)[0][0]
        mask[first_true_loc] = False
        df_joined_tokens = df_joined_tokens.loc[:, ~mask]
        
        
        # drop the second column that is also called "address", keep the first
        # pop columns called address


        address_df = df_joined_tokens.pop("address")

        df_joined_tokens["address"] = address_df.iloc[:, 0]
        
        
        block_id_df = df_joined_tokens.pop("block_id")
        df_joined_tokens["block_id"] = block_id_df.iloc[:, 0]
        """
        # remove other duplicates
        df_joined_tokens = df_joined_tokens.loc[:, ~df_joined_tokens.columns.duplicated(keep="first")]


        self.disconnect_databases()

        results = (df_joined, df_joined_tokens, df_tokens, df_prices)
        compress_pickle(cache_path, results)

        return df_joined, df_joined_tokens, df_tokens, df_prices


    @log_time
    def get_transactions_with_botstatus_and_prices(self, prediction_file):

        df_joined, df_joined_tokens, df_tokens, df_prices = self.get_transactions_with_botstatus_and_prices_SQL(prediction_file)


        mapping_contract_to_symbol = dict(zip(df_tokens["address"].str.lower(), df_tokens["Symbol"]))
        df_joined_tokens["Symbol"] = df_joined_tokens["address"].map(mapping_contract_to_symbol)

        # convert df_prices to a dictionary block_id maps to eth_usd
        prices = df_prices.set_index("block_id")

        prices_d = prices["eth_usd"]
        prices_d = prices_d.to_dict()

        df_joined["eth_usd"] = df_joined["block_id"].map(prices_d) / 10 ** 18
        df_joined["ETH Value (USD)"] = df_joined["value"] * df_joined["eth_usd"]

        tokens_not_usd = df_tokens[df_tokens.dollarequivalent == 0]
        addresses = [x.lower() for x in tokens_not_usd["address"]]
        pairs = [f"{x.lower()}_usd" for x in tokens_not_usd["Symbol"]]
        decimal_factors = [10 ** x for x in tokens_not_usd["decimals"]]

        for pair in pairs:
            df_joined_tokens[pair] = np.nan
            df_joined_tokens[f"{pair.split('_')[0].upper()} Value (USD)"] = np.nan

        for pair, decimal, address in zip(pairs, decimal_factors, addresses):
            prices_d = prices[pair]
            prices_d = prices_d.to_dict()
            mask = address == df_joined_tokens.address
            df_subset = df_joined_tokens[mask]


            df_joined_tokens[pair][mask] = df_subset["block_id"].map(prices_d)/ decimal
            df_joined_tokens[f"{pair.split('_')[0].upper()} Value (USD)"][mask] = df_joined_tokens[pair][mask] * df_subset["value"]

        tokens_usd = df_tokens[df_tokens.dollarequivalent == 1]
        addresses = [x.lower() for x in tokens_usd["address"]]
        pairs = [f"{x.lower()}_usd" for x in tokens_usd["Symbol"]]
        decimal_factors = [10 ** x for x in tokens_usd["decimals"]]


        for pair in pairs:
            df_joined_tokens[pair] = np.nan
            df_joined_tokens[f"{pair.split('_')[0].upper()} Value (USD)"] = np.nan


        for pair, decimal, address in zip(pairs, decimal_factors, addresses):


            mask = address == df_joined_tokens.address
            df_subset = df_joined_tokens[mask]


            df_joined_tokens[pair][mask] = 1 / decimal
            df_joined_tokens[f"{pair.split('_')[0].upper()} Value (USD)"][mask] = df_joined_tokens[pair][mask] * df_subset["value"]

        df_joined_tokens["Token Value (USD)"] = df_joined_tokens[[x for x in df_joined_tokens.columns if "Value (USD)" in x]].sum(axis=1)
        return df_joined, df_joined_tokens


    @log_time
    def get_cumulative_value(self, prediction_file):
        df_joined, df_joined_tokens = self.get_transactions_with_botstatus_and_prices(prediction_file)

        # group by bot (0 or 1) and calculate the value for each block
        value_bots = df_joined.groupby(["bot", "block_id"])["ETH Value (USD)"].sum()
        bot_series = value_bots[True]
        non_bot_series = value_bots[False]

        min_block, max_block = df_joined["block_id"].min(), df_joined["block_id"].max()
        blocks = np.arange(min_block, max_block + 1)
        value_df = pd.DataFrame({"bot": bot_series, "non_bot": non_bot_series}, index=blocks)

        value_df = value_df.fillna(0)
        ################################################
        # INVESTIGATION WHY THERE IS SO LITTLE DIFFERENCE IN THE CUMULATIVE VALUE BUT SUCH BIG DIFF IN #TX and #ADDR
        #predictions_randomforest_results_keep_scaling = load_prediction_results("randomforest_results_keep_scaling", self.prefix)
        #predictions_randomforest_results_new_scaling = load_prediction_results("randomforest_results_new_scaling", self.prefix)
        #
        # i want the difference between the two predictions, namely: the ones that were predicted by new but not by keep
        # both have the same index and the same addresses but the difference is in bot 0/1
        # so i can just subtract the two columns
        #mask = (predictions_randomforest_results_new_scaling["bot"] - predictions_randomforest_results_keep_scaling["bot"]) == 1 # change this sto 1 to get the other direction
        # get the addresses
        #addresses = predictions_randomforest_results_new_scaling[mask]["address"]
        #df_joined_usd_only = df_joined[["address", "ETH Value (USD)"]]
        #df_joined_usd_only.index = df_joined_usd_only["address"]
        #df_joined_usd_only = df_joined_usd_only.loc[addresses]
        #df_joined_usd_only.drop(columns=["address"], inplace=True)
        #df_joined_usd_only = df_joined_usd_only.groupby("address").sum()
        #print(df_joined_usd_only.sum())

        # Conclusion for largesample4 (=2023 window), we find 45833 addresses that are bots in the main-window preprocessing
        # but not in the window-specific preprocessing. The difference in value is about 2.3B USD
        # On the other hand, we find 890958 addresses (about 20x as many) that are bots in the window-specific preprocessing
        # but not in the main-window preprocessing. The difference in value is about 2.5B USD so we see it is about the same
        # and therefore we see little difference in the cumulative plots
        ########################################################################
        # Calculate the cumulative sum for both columns
        value_df['cumulative_bot'] = value_df['bot'].cumsum()
        value_df['cumulative_non_bot'] = value_df['non_bot'].cumsum()
        cumulative_value_df = value_df[['cumulative_bot', 'cumulative_non_bot']]

        if df_joined_tokens.empty:
            return cumulative_value_df, None, None

        path_to_tokens = f"{self.prefix}/data_lightweight/token_addresses.csv"
        df_tokens = pd.read_csv(path_to_tokens)
        symbols = df_tokens["Symbol"]
        cols = [f"{x} Value (USD)" for x in symbols]

        value_bots_tokens = df_joined_tokens.groupby(["bot", "block_id"])[["Token Value (USD)"] + cols].sum()

        # just slicing with [True] or .loc[True] doesnt work with multiple columns for whatever reason! workaround:
        bot_df_tokens = value_bots_tokens.loc[value_bots_tokens.index.get_level_values('bot') == True]
        bot_df_tokens.index = bot_df_tokens.index.droplevel(0)
        non_bot_df_tokens =  value_bots_tokens.loc[value_bots_tokens.index.get_level_values('bot') == False]
        non_bot_df_tokens.index = non_bot_df_tokens.index.droplevel(0)

        min_block_tokens, max_block_tokens = df_joined_tokens["block_id"].min(), df_joined_tokens["block_id"].max()
        blocks_tokens = np.arange(min_block_tokens, max_block_tokens + 1)
        # concat
        value_df_tokens_bot = pd.DataFrame(bot_df_tokens, index=blocks_tokens)
        value_df_tokens_non_bot = pd.DataFrame(non_bot_df_tokens, index=blocks_tokens)
        value_df_tokens_bot = value_df_tokens_bot.fillna(0)
        value_df_tokens_non_bot = value_df_tokens_non_bot.fillna(0)

        value_df_tokens_bot = value_df_tokens_bot.cumsum()
        value_df_tokens_non_bot = value_df_tokens_non_bot.cumsum()

        # NOT TESTED YET:
        self.logger.info(f"Total Ether value in USD of bots and non-bots: {df_joined.groupby('bot')['ETH Value (USD)'].sum()}")
        self.logger.info(f"Total Token value in USD of bots and non-bots: {df_joined_tokens.groupby('bot')['Token Value (USD)'].sum()}")


        return cumulative_value_df, value_df_tokens_bot, value_df_tokens_non_bot

    @log_time
    def calculate_aggregate_stats_grouped_by_bot_status(self, prediction_file):

        df_joined, df_joined_tokens = self.get_transactions_with_botstatus_and_prices(prediction_file)

        n_bot_tx = df_joined["bot"].sum()
        percentage_bot_tx = n_bot_tx / len(df_joined)

        n_bots = df_joined[df_joined["bot"] == True]["from_address"].nunique()
        n_addresses = df_joined["from_address"].nunique()
        percentage_bots = n_bots / n_addresses


        df_joined["sendstoCA"] = df_joined["account_type_to_address"] == "CA"
        mask = df_joined["bot"] == 1
        average_smart_contract_interaction_bot = df_joined[mask]["sendstoCA"].mean()
        average_smart_contract_interaction_non_bot = df_joined[~mask]["sendstoCA"].mean()



        return_dict = {
            "n_bot_tx": n_bot_tx,
            "percentage_bot_tx": percentage_bot_tx,
            "n_bots": n_bots,
            "n_addresses": n_addresses,
            "percentage_bots": percentage_bots,
            "average_sci_bot": average_smart_contract_interaction_bot,
            "average_sci_non_bot": average_smart_contract_interaction_non_bot
        }

        return return_dict

        # more statistics




if __name__ == "__main__":

    an = Analysis(load_configs(".."), prefix="..")
    top_setup = an.create_cluster_table_clustersizefixed()
