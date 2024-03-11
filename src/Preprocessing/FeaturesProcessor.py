from src.Datamodels.PipelineComponent import PipelineComponent
import pandas as pd
from sklearn.mixture import GaussianMixture
import umap
from sklearn import preprocessing
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from ..Aggregating.features import entropy_from_counts_remove_zeros
import time

class NAFiller:

    def __init__(self, type):
        self.type = type

    def fit(self, df: pd.DataFrame):
        if self.type == "mean":
            self.means = df.mean()
        elif self.type == "median":
            self.means = df.median()
        elif self.type == "-1":
            self.means = -1
        else:
            raise NotImplementedError(f"Type {self.type} not implemented")

    def transform(self, df: pd.DataFrame):
        return df.fillna(self.means)

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)


class FeaturesModeller(PipelineComponent):

    # def init, use params of parent class and additionally: nafill: str
    # the parent class has config and prefix
    def __init__(self, configs, prefix, nafill_type):
        super().__init__(configs, prefix)
        self.nafill_type = nafill_type

    def na_fill_mean(self, df: pd.DataFrame, means):
        """
        Fill missing values with the mean of the column
        """
        return df.fillna(means)

    def load_eval_set(self):
        path = f"{self.prefix}/data_lightweight/vote_detailed.csv"
        # make index lower case
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.str.lower()
        return df

    def preprocess(self, df_features: pd.DataFrame, desired_cols=None) -> pd.DataFrame:
        """
        This preprocessing only reduces the dimensionality and cleans the data. This is not preprocessing for ML models like normalization
        :param df_features:
        :return:
        """

        df_features = df_features.fillna(np.nan)

        if desired_cols is not None:
            df_features = df_features.reindex(columns=desired_cols, fill_value=np.nan) # untested

        # should be moved to feature extraction but is here for now
        # get address__0 to address__f columns and calculate entropy
        address_cols = [col for col in df_features.columns if col.startswith("address__") and len(col) == 10]
        address_df = pd.concat([df_features.pop(x) for x in address_cols], axis=1)
        df_features["address_entropy"] = address_df.apply(entropy_from_counts_remove_zeros, axis=1)

        # convert column to float if possible, can be dropped later TODO recalculate
        for col in df_features.columns:
            try:
                df_features[col] = df_features[col].astype(float)
            except:
                pass

        ### dimensionality reduction
        # check how sparse this part of the data is

        # drop columns
        #  they pollute the tvc and benfords law clustering, so we drop these features
        # if it is outmin or inmax, they are calculated based on the constant product formula, which makes them appear random
        drop_words = ["amountoutmin", "amountinmax"]
        drop_cols = [col for col in df_features.columns if any(word in col for word in drop_words)]
        df_features = df_features.drop(columns=drop_cols)

        swap_cols = [col for col in df_features.columns if col.startswith("scb__sb") or col.startswith("scb__eb__swap")]

        cols = df_features.columns
        cols_samename = [f"{col.split('_')[0]}|{col.split('__')[-1]}" for col in cols if col.startswith("scb")]
        cols_samename_filtered = [col for col in cols_samename if col.startswith("scb")]
        cols_realname_filtered = df_features.columns[df_features.columns.isin(cols_samename_filtered)]
        subset = df_features[cols_realname_filtered].transpose()
        subset["colnamereduced"] = cols_samename_filtered

        scb_cols_end = [col.split('__')[-1] for col in cols if
                        col.startswith("scb__eb__swap") or col.startswith("scb__sb")]
        endings = list(set(scb_cols_end))

        value_based_means = {}
        path_based_means = {}

        for ending in endings:
            value_based_means["scb__value__" + ending] = df_features.filter(
                regex=f"(scb__eb__swap|scb__sb).*__.*amount.*{ending}$").mean(axis=1, skipna=True)
            path_based_means["scb__pathlength__" + ending] = df_features.filter(
                regex=f"(scb__eb__swap|scb__sb).*__.*path.*{ending}$").mean(axis=1, skipna=True)

        df_value_reduced = pd.DataFrame(value_based_means).drop(columns=["scb__value__swaps_per_block"], errors="ignore")
        df_path_reduced = pd.DataFrame(path_based_means).drop(columns=["scb__pathlength__swaps_per_block",
                                                                       "scb__pathlength__benfords",
                                                                       "scb__pathlength__tvc"], errors="ignore")

        df_features.drop(swap_cols, axis=1, inplace=True)
        df_features = pd.concat([df_features, df_value_reduced, df_path_reduced], axis=1)

        reduced_sparsity = df_features["scb__pathlength__max"].isna().sum() / len(df_features)

        ### NA columns
        number_nonNA = df_features.isna().__invert__().sum(axis=0)
        mask = number_nonNA == 0
        print(df_features.columns[mask])

        print(f"sparsity after dimensionality reduction: {reduced_sparsity}")
        df_features.drop(["tx__generic__receipt_status_1"], axis=1, inplace=True)
        print(
            "Dropped tx__generic__receipt_status_1 because it is the same information as tx__generic__receipt_status_0. If one is 1 the other is 0")

        # sort columns alphabetically
        df_features = df_features.reindex(sorted(df_features.columns), axis=1)
        return df_features

    def fit_transform_normalize_df(self, df):
        self.min_max_scaler = preprocessing.MinMaxScaler()
        df_normalized = pd.DataFrame(self.min_max_scaler.fit_transform(df), columns=df.columns,
                                     index=df.index)
        return df_normalized

    def transform_normalize_df(self, df):
        df_normalized = pd.DataFrame(self.min_max_scaler.transform(df), columns=df.columns,
                                     index=df.index)
        return df_normalized
    def fit_transform_UMAP_embedding(self, df_normalized, dimensions=2):
        self.umap_dimensions = dimensions
        self.UMAP_nafiller = NAFiller(self.nafill_type)
        df_nafilled = self.UMAP_nafiller.fit_transform(df_normalized)
        self.reducer = umap.UMAP(n_components=self.umap_dimensions)
        embedding = self.reducer.fit_transform(df_nafilled)
        df_embedding = pd.DataFrame(embedding, index=df_normalized.index, columns=["x", "y", "z"][0:self.umap_dimensions])
        return df_embedding

    def transform_UMAP_embedding(self, df_normalized):
        df_nafilled = self.UMAP_nafiller.transform(df_normalized)
        embedding = self.reducer.transform(df_nafilled)
        df_embedding = pd.DataFrame(embedding, index=df_normalized.index, columns=["x", "y", "z"][0:self.umap_dimensions])
        return df_embedding

    def fit_transform_BIC_optimal_GMM(self, df_normalized):
        self.GMM_nafiller = NAFiller(self.nafill_type)
        df_nafilled = self.GMM_nafiller.fit_transform(df_normalized)

        # Perform clustering with gaussian mixture models
        df_nafilled_sampled = df_nafilled.sample(min(30_000, len(df_normalized)))
        scores = []
        options = list(range(2, 31, 2))
        for i in tqdm(options):
            print(i)
            gm = GaussianMixture(n_components=i, random_state=0).fit(df_nafilled_sampled)
            scores.append(gm.bic(df_nafilled_sampled))

        plt.plot(scores)
        plt.show()

        scores_arr = np.array(scores)
        # min score
        min_score = options[scores_arr.argmin()]
        self.logger.debug(f"number of clusters with min BIC: {min_score}")
        gm = GaussianMixture(n_components=min_score, random_state=0).fit(df_nafilled)
        print(gm.n_components)
        self.gm = gm

    def fit_GMM(self, df_normalized, n_clusters):
        self.GMM_nafiller = NAFiller(self.nafill_type)
        df_nafilled = self.GMM_nafiller.fit_transform(df_normalized)
        gm = GaussianMixture(n_components=n_clusters, random_state=0).fit(df_nafilled)
        self.gm = gm

    def transform_GMM(self, df_normalized):
        df_nafilled = self.GMM_nafiller.transform(df_normalized)
        transformed = self.gm.predict(df_nafilled)
        df = pd.DataFrame(transformed, index=df_nafilled.index, columns=["cluster"])
        return df

    def transform_BIC_optimal_GMM(self, df_normalized):

        df_nafilled = self.GMM_nafiller.transform(df_normalized)
        transformed = self.gm.predict(df_nafilled)
        df = pd.DataFrame(transformed, index=df_nafilled.index, columns=["cluster"])
        return df

    def get_GMM_n_clusters(self):
        return self.gm.n_components


    def fit_transform_KMEANS(self, df_normalized):
        self.XMEAN_nafiller = NAFiller(self.nafill_type)
        df_nafilled = self.XMEAN_nafiller.fit_transform(df_normalized)
        df_nafilled_sampled = df_nafilled.sample(min(30_000, len(df_normalized)))
        X_sampled = df_nafilled_sampled.values
        init_clust, max_clust = 1, 15 #  wont get over 15 anyway as ive seen

        import numpy as np

        def calculate_bic(kmeans, X):
            # Number of data points
            n = X.shape[0]

            # Number of clusters
            m = kmeans.n_clusters

            # Compute the sum of the squared distances from each point to its assigned center
            centroids = kmeans.cluster_centers_
            cluster_labels = kmeans.labels_
            sum_of_squares = np.sum([np.linalg.norm(X[i] - centroids[cluster_labels[i]]) ** 2 for i in range(n)])

            # Compute the BIC
            bic = np.log(n) * m + n * np.log(sum_of_squares / n)

            return bic

        print("optimal clusters:")
        start = time.time()

        from sklearn.cluster import KMeans

        bics = []
        for k in tqdm(range(init_clust, max_clust)):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(X_sampled)
            bics.append(calculate_bic(kmeans, X_sampled))

        from kneed import KneeLocator
        n_clusts = np.array(range(init_clust, max_clust))
        # normalize bics to 0 1
        bics = np.array(bics)
        bics = (bics - bics.min()) / (bics.max() - bics.min())
        kneedle = KneeLocator(n_clusts, -bics, S=1.0, curve="concave", direction="increasing")
        n_opt = kneedle.knee
        print(n_opt)

        # use kmeans to get the cluster centers
        X = df_nafilled.values
        kmeans = KMeans(n_clusters=n_opt, random_state=0).fit(X)
        self.kmeans = kmeans

    def get_KMEANS_n_clusters(self):
        return self.kmeans.n_clusters


    def transform_XMEANS(self, df_normalized):
        df_nafilled = self.XMEAN_nafiller.transform(df_normalized)
        transformed = self.xmeans_instance.predict(df_nafilled.values)
        df = pd.DataFrame(transformed, index=df_nafilled.index, columns=["cluster"])
        return df

    def transform_KMEANS(self, df_normalized):
        df_nafilled = self.XMEAN_nafiller.transform(df_normalized)
        transformed = self.kmeans.predict(df_nafilled.values)
        df = pd.DataFrame(transformed, index=df_nafilled.index, columns=["cluster"])
        return df

    def fit_KMEANS(self, df_normalized, n_cluster):
        self.XMEAN_nafiller = NAFiller(self.nafill_type)
        df_nafilled = self.XMEAN_nafiller.fit_transform(df_normalized)

        X = df_nafilled.values

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X)
        self.kmeans = kmeans


    def get_XMEANS_n_clusters(self):
        return len(self.xmeans_instance.get_centers())

    def pickle(self):
        with open(f"{self.prefix}/models/modeller_{self.configs['General']['run_name']}.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(self):
        with open(f"{self.prefix}/models/modeller_{self.configs['General']['run_name']}.pkl", "rb") as f:
            return pickle.load(f)
