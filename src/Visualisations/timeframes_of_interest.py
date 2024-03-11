from src.Datamodels.Visualisation import Visualisation
import matplotlib.pyplot as plt
import pandas as pd
import datetime

class Vis(Visualisation):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def create_visualisation(self):
        # TODO not tested yet
        df_price = self.load_data()
        plt.figure(figsize=(10, 5))
        df_price["High"].plot()

        dates_blocks = [
            ((datetime.datetime(2018, 1, 4, 0, 0), datetime.datetime(2018, 1, 22, 0, 0))),#, 4850000),
            ((datetime.datetime(2019, 10, 8, 0, 0), datetime.datetime(2019, 10, 24, 0, 0))),#, 8700000),
            ((datetime.datetime(2021, 10, 27, 0, 0), datetime.datetime(2021, 11, 12, 0, 0))),#, 13500000),
            ((datetime.datetime(2022, 9, 9, 0, 0), datetime.datetime(2022, 9, 24, 0, 0))),#, 15500000),
            ((datetime.datetime(2023, 8, 12, 0, 0), datetime.datetime(2023, 8, 26, 0, 0))),#, 17900000),
        ]
        window_names_depicted = ["largesample1", "largesample2", "largesample3", "large", "largesample4"]


        window_names_to_show = self.configs["General"]["window_names"]
        window_names_internal = self.configs["General"]["window_names_internal"]
        mapping = dict(zip(window_names_internal, window_names_to_show))

        timeframe_names = [mapping[name] for name in window_names_depicted]
        combined = zip(dates_blocks, timeframe_names)

        for (start, end), text in combined:
            # plot marker for the area of each timeframe of interest
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)

            # Plot marker for the area of each timeframe of interest
            plt.axvspan(start, end, color="red", alpha=0.2)

            # Add block number label inside each shaded region
            plt.text(start + (end - start) / 2, df_price["High"].max() - 3500, text, ha="center", va="bottom",
                     rotation=45)

        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        #plt.title("ETH Price and Observation Windows")


if __name__ == "__main__":
    v = Vis()
    prefix = "../.."
    v.prefix = prefix
    v.render_visualisation()