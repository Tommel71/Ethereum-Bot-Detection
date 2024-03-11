import shap
from src.Datamodels.Visualisation import Visualisation
import matplotlib.pyplot as plt
import numpy as np

class Vis(Visualisation):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def create_visualisation(self):
        df = self.load_data()

        plt.figure(figsize=(10, 5))
        df_to_plot = df[["type", "tx__generic__gas_min"]]

        # Common bins for all histograms
        bins = np.histogram_bin_edges(np.log10(df_to_plot["tx__generic__gas_min"]), bins=25)

        # Creating a stacked histogram plot
        plt.figure(figsize=(10, 5))

        plt.hist([np.log10(df_to_plot[df_to_plot["type"] == "Arbitrage"]["tx__generic__gas_min"]),
                  np.log10(df_to_plot[df_to_plot["type"] == "Sandwich"]["tx__generic__gas_min"]),
                  np.log10(df_to_plot[df_to_plot["type"] == "Liquidation"]["tx__generic__gas_min"]),
                  np.log10(df_to_plot[df_to_plot["type"] == "non-MEV"]["tx__generic__gas_min"])],
                 bins=bins, stacked=True, alpha=0.7,
                 label=["Arbitrage", "Sandwich", "Liquidation", "non-MEV"])

        plt.legend(loc='upper right')
        plt.xlabel("Mean GasLimit (log)")
        plt.ylabel("Frequency")

        plt.tight_layout()

        """
                
        df = self.load_data()

        plt.figure(figsize=(10, 5))
        df_to_plot = df[["type", "tx__generic__gas_min"]]

        # Common bins for all histograms
        bins = np.histogram_bin_edges(np.log10(df_to_plot["tx__generic__gas_min"]), bins=25)

        plt.hist(np.log10(df_to_plot[df_to_plot["type"] == "Arbitrage"]["tx__generic__gas_min"]), bins=bins, alpha=0.5,
                 label="Arbitrage")
        plt.hist(np.log10(df_to_plot[df_to_plot["type"] == "Sandwich"]["tx__generic__gas_min"]), bins=bins, alpha=0.5,
                 label="Sandwich")
        plt.hist(np.log10(df_to_plot[df_to_plot["type"] == "Liquidation"]["tx__generic__gas_min"]), bins=bins,
                 alpha=0.5, label="Liquidation")
        plt.hist(np.log10(df_to_plot[df_to_plot["type"] == "non-MEV"]["tx__generic__gas_min"]), bins=bins, alpha=0.5,
                 label="non-MEV")

        plt.legend(loc='upper right')
        plt.xlabel("Mean GasLimit (log)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        """

if __name__ == "__main__":
    prefix = "../.."
    v = Vis()
    v.prefix = prefix
    v.render_visualisation()