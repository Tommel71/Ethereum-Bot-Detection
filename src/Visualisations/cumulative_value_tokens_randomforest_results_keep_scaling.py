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
        # Set up the facet grid
        custom_palette = sns.color_palette("tab10")

        data_dict = {}
        agg_col = "Token Value (USD)"
        for key, dfs in data.items():
            df_bot, df_nonbot = dfs[0], dfs[1]
            agg_cols_of_dfs = [df_bot[agg_col], df_nonbot[agg_col]]
            df = pd.concat(agg_cols_of_dfs, axis=1)
            df.columns = ["cumulative_bot", "cumulative_non_bot"]
            data_dict[key] = df

        names = self.configs["General"]["window_names"]
        assert list(data_dict.keys()) == ['largesample1', 'largesample2', 'largesample3', 'largesample4', 'large']
        df_list = list(data_dict.values())

        [df.reset_index(inplace=True, drop=True) for df in df_list]
        [df.reset_index(inplace=True) for df in df_list]
        [df.rename(columns={"index": "Blocks"}, inplace=True) for df in df_list]
        g = sns.FacetGrid(col='DataFrame',
                          data=pd.concat([df.assign(DataFrame=names[i]) for i, df in enumerate(df_list)]),
                          col_wrap=3,
                          height=3.33
                          )

        # Plot each DataFrame
        g.map(plt.plot, 'Blocks', 'cumulative_bot', label='Bot', color=custom_palette[0])
        g.map(plt.plot, 'Blocks', 'cumulative_non_bot', label='Human', color=custom_palette[1])

        # Customize plot labels and titles
        g.set_axis_labels('Blocks', 'Value (USD)')
        g.set_titles('{col_name}')

        # Apply the custom y-axis label formatter
        # Determine the maximum y-value in your data
        max_y_value = max(df[['cumulative_bot', "cumulative_non_bot"]].max().max() for df in df_list)

        # Calculate the scale and create 6 ticks
        scale = 10 ** np.floor(np.log10(max_y_value))
        n_ticks = int(int(max_y_value) // scale + 1)
        yticks = [i * scale for i in range(n_ticks)]  # 0, 1*scale, 2*scale, ..., 6*scale
        # TODO B is not always the right unit, could be K M
        yticklabels = [f'{i} B' for i in range(n_ticks)]

        # Set y-axis ticks and labels
        g.set(yticks=yticks, yticklabels=yticklabels)
        g.fig.suptitle('Cumulative Value over Time')

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
        'cumulative_non_bot': np.cumsum(np.random.randn(100))  # Random cumulative values for human
    }
    df = pd.DataFrame(data)
    cumulative_value_df_list.append(df)

# Now you have a list of five sample DataFrames for testing


    """