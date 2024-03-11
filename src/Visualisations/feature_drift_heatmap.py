import seaborn as sns
from src.Datamodels.Visualisation import Visualisation
import matplotlib.pyplot as plt
from tools import feature_name_mapper
from tools import get_window_names_mapping

class Vis(Visualisation):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def create_visualisation(self):

        ddd_norm, settingnames_small = self.load_data()
        # drop rows of dd_norm containing type_0
        ddd_norm = ddd_norm[~ddd_norm.index.str.contains("type")]
        ddd_norm = ddd_norm[~ddd_norm.index.str.contains("pathlength")]

        # ignore nans
        maxes = ddd_norm.max(axis=1, skipna=True)
        top_5 = maxes.nlargest(5)
        mins = ddd_norm.min(axis=1, skipna=True)
        bottom_5 = mins.nsmallest(5)

        window_names_mapping = get_window_names_mapping(self.configs)


        names_of_interest = list(top_5.index) + list(bottom_5.index)

        # create a heatmap of the top features
        heatmap_data = ddd_norm.loc[names_of_interest][settingnames_small].rename(columns=window_names_mapping)
        #vmin, vmax = heatmap_data.min().min(), heatmap_data.max().max()
        heatmap_data.index = feature_name_mapper(heatmap_data.index)

        renaming = {
            "Tx-GasPriceMin": "TX-GasPrice-Min",
            "Tx-GasPriceMedian": "TX-GasPrice-Median",
            "Tx-GasPriceMean": "TX-GasPrice-Mean",
            "Tx-Time-InTransactionFrequency": "In-TX-Frequency",
            "Tx-GasPriceQuantile95": "TX-GasPrice-Quantile95",
            "Scb-Value-Benfords": "Token-TX-Value-Benfords",
            "Tx-Time-InMax": "In-TX-Time-Max",
            "Tx-Time-InQuantile95": "In-TX-Time-Quantile95",
            "Tx-Time-InMean": "In-TX-Time-Mean",
            "Tx-Time-InMedian": "In-TX-Time-Median",
        }

        heatmap_data = heatmap_data.rename(index=renaming)

        # plot

        plt.figure(figsize=(10, 5))
        # title
        #plt.title("Feature Drift Heatmap")
        scale_difference = heatmap_data.max().max()/ -heatmap_data.min().min()
        # scale up the negative values and then format them so that they show the right value
        mask_neg = heatmap_data < 0
        #heatmap_data[mask_neg] = heatmap_data[mask_neg] * scale_difference
        fmt_func = lambda x: f"{x:.2f}" if x > 0 else f"{x/scale_difference:.2f}"
        annot = heatmap_data.applymap(fmt_func)

        sns.heatmap(heatmap_data, annot=annot, fmt="", cmap="RdBu_r", center=0)


if __name__ == "__main__":
    prefix = "../.."
    v = Vis()
    v.prefix = prefix
    v.render_visualisation()