from src.Datamodels.Visualisation import Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FuncFormatter
import numpy as np

class Vis(Visualisation):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def create_visualisation(self):

        data = self.load_data()
        data2_path = f"{self.prefix}/outputs/{self.runs[0]}/figtables_pickled/{self.chapter}/cumulative_value_tokens_randomforest_results_new_scaling.pkl"
        data_tokens = pd.read_pickle(data2_path)
        data_dict = {}
        agg_col = "Token Value (USD)"
        for key, dfs in data_tokens.items():
            df_bot, df_nonbot = dfs[0], dfs[1]
            agg_cols_of_dfs = [df_bot[agg_col], df_nonbot[agg_col]]
            df = pd.concat(agg_cols_of_dfs, axis=1)
            df.columns = ["cumulative_bot", "cumulative_non_bot"]
            data_dict[key] = df

        data_tokens = data_dict
        # Set up the facet grid
        custom_palette = sns.color_palette("tab10")

        # from seaborn code:
        # Calculate the base figure size
        # This can get stretched later by a legend
        #figsize = (ncol * height * aspect, nrow * height)
        # aspect is 1 per default; we want our standard width of 10. Therefore, we pick height

        # but the legend takes up
        # height = 10 / (ncol * aspect) = 3.33


        # Define a function to format y-axis labels
        def y_axis_formatter(x, pos):
            return f"{x / 1e9:.0f} B"

        names = self.configs["General"]["window_names"]
        assert list(data.keys()) == ['largesample1', 'largesample2', 'largesample3', 'largesample4', 'large']
        assert list(data_tokens.keys()) == ['largesample1', 'largesample2', 'largesample3', 'largesample4', 'large']


        df_list = list(data.values())
        [df.reset_index(inplace=True, drop=True) for df in df_list]
        [df.reset_index(inplace=True) for df in df_list]
        [df.rename(columns={"index": "Blocks"}, inplace=True) for df in df_list]


        df_list_tokens = list(data_tokens.values())
        [df.reset_index(inplace=True, drop=True) for df in df_list_tokens]
        [df.reset_index(inplace=True) for df in df_list_tokens]
        [df.rename(columns={"index": "Blocks"}, inplace=True) for df in df_list_tokens]


        # add columns to df_list from df_list_tokens. Same name but _with_tokens
        for i, df in enumerate(df_list):
            df["cumulative_bot_with_tokens"] = df_list_tokens[i]["cumulative_bot"] + df["cumulative_bot"]
            df["cumulative_non_bot_with_tokens"] = df_list_tokens[i]["cumulative_non_bot"] + df["cumulative_non_bot"]


        g = sns.FacetGrid(col='DataFrame',
                          data=pd.concat([df.assign(DataFrame=names[i]) for i, df in enumerate(df_list)]),
                          col_wrap=3,
                          height=3.33
                          )

        # Plot each DataFrame
        g.map(plt.plot, 'Blocks', 'cumulative_bot', label='Bot TX-Only', color=custom_palette[0])
        g.map(plt.plot, 'Blocks', 'cumulative_bot_with_tokens', label='Bot with Tokens', color=custom_palette[0], linestyle="dashed")
        g.map(plt.plot, 'Blocks', 'cumulative_non_bot', label='Human TX-Only', color=custom_palette[1])
        g.map(plt.plot, 'Blocks', 'cumulative_non_bot_with_tokens', label='Human with Tokens', color=custom_palette[1], linestyle="dashed")

        # Customize plot labels and titles
        g.set_axis_labels('Blocks', 'Value (USD)')
        g.set_titles('{col_name}')

        # Apply the custom y-axis label formatter
        # Determine the maximum y-value in your data
        max_y_value = max(df[["cumulative_bot_with_tokens", "cumulative_non_bot_with_tokens"]].max().max() for df in df_list)

        # Calculate the scale and create 6 ticks
        # Calculate the scale and create 6 ticks
        scale = 10 ** np.floor(np.log10(max_y_value))
        n_ticks = int(int(max_y_value) // scale + 2)  # one for 0 and one for the one above the max val
        yticks = [i * scale for i in range(n_ticks)]  # 0, 1*scale, 2*scale, ..., 6*scale

        def calculate_y_ticklabels(scale, n_ticks):
            shorthandmapping = {
                0: "",
                1000: "K",
                1000000: "M",
                1000000000: "B",
                1000000000000: "T",
            }

            # pick shorthand
            for i in list(shorthandmapping.keys())[::-1]:
                if scale >= i:
                    denom = i
                    break

            shorthand = shorthandmapping[denom]
            rest = int(scale / denom)
            yticklabels = [f'{i * rest} {shorthand}' for i in range(n_ticks)]  # 0, 1*scale, 2*scale, ..., 6*scale

            return yticklabels

        yticklabels = calculate_y_ticklabels(scale, n_ticks)

        # Set y-axis ticks and labels
        g.set(yticks=yticks, yticklabels=yticklabels)
        #g.fig.suptitle('Cumulative Value transferred over Time')

        # Add legends and grids
        g.tight_layout()
        g.add_legend(bbox_to_anchor=(0.77, 0.25), ncol=1)


if __name__ == "__main__":
    prefix = "../.."
    v = Vis()
    v.prefix = prefix
    v.render_visualisation()

    """
import pandas as pd
import numpy as np

# Create a list to store the DataFrames
cumulative_value_df_list = []

# Generate five sample DataFrames
for i in range(5):
    # Create a DataFrame with random values
    data = {
        'index': range(1, 101),  # Assuming 100 data points
        'cumulative_bot': np.cumsum(np.random.randn(100)),  # Random cumulative values for bot
        'cumulative_non_bot': np.cumsum(np.random.randn(100))  # Random cumulative values for non-bot
    }
    df = pd.DataFrame(data)
    cumulative_value_df_list.append(df)

# Now you have a list of five sample DataFrames for testing

    
    """