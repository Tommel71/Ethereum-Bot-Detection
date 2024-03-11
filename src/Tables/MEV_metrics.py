from src.Datamodels.Table import Table
import pandas as pd

class Tab(Table):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def set_outnames(self):
        self.outnames = [self.name]

    def create_tex_code(self):
        df = self.load_data()
        # capitalize index
        map = {"Arbitrage": "Binary Arbitrage", "Liquidation": "Binary Liquidation", "Sandwich": "Binary Sandwich"}
        # make sure values are in index
        assert set(map.keys()).issubset(set(df.index))
        df.index = df.index.map(map)
        # rename index to "Dataset"
        df.index.name = "Dataset"

        # replace substring nan with 1.00
        df["Precision"] = df["Precision"].astype(str).str.replace("nan", "1.00")

        tex_string = df.to_latex(index=True)
        return tex_string


if __name__ == "__main__":
    prefix = "../.."
    v = Tab()
    v.prefix = prefix
    v.create_and_save()