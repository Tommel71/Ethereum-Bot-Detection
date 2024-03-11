from src.Datamodels.Visualisation import Visualisation
import matplotlib.pyplot as plt
import pandas as pd

class Vis(Visualisation):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def set_outfolder(self):
        self.outfolder = f"{self.prefix}/paper_assets/figures"


    def create_visualisation(self):
        dfs, features = self.load_data()

        plt.figure(figsize=(10, 5))

        max_recall_idx = dfs[0].idxmax()["Recall"]
        max_idx_point = dfs[0].loc[max_recall_idx]

        for df in dfs:
            # add a 0,0 point in the beginning
            df = pd.concat([pd.DataFrame({"Recall": [0], "Precision": [0]}), df])
            df.drop_duplicates(inplace=True)
            # remove Recall = 1
            df = df[df["Recall"] < 1]
            plt.plot(df["Recall"], df["Precision"])

        # add max point with white border
        plt.plot(max_idx_point["Recall"], max_idx_point["Precision"], marker="o", color="red", markeredgecolor="white")
        plt.annotate(f"Only Bots", (max_idx_point["Recall"] -0.1, max_idx_point["Precision"]-0.1))

        # TODO HARDCODED
        top_x, top_y = 0.78, 0.88
        plt.plot(top_x, top_y, marker="x", color="blue")
        plt.annotate(f"Random Forest", (top_x - 0.1, top_y - 0.075))


        renaming = {
            "tx__time__outtime_hourly_entropy": "Out-TX-Entropy",
            "tx__custom__n_tx_per_block": "Out-TX-Per-Block",
            "tx__time__outtime_transaction_frequency": "Out-TX-Frequency",
            "tx__generic__gas_price_max": "TX-GasPrice-Max",
            "tx__time__outtime_sleepiness": "Out-TX-Sleepiness"
        }
        # rename features
        features = [renaming.get(feature, feature) for feature in features]


        plt.legend(features)
        #plt.title(f"Precision Recall Curves for select Features")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        #plt.show()


if __name__ == "__main__":
    v = Vis()
    prefix = "../.."
    v.prefix = prefix
    v.render_visualisation()