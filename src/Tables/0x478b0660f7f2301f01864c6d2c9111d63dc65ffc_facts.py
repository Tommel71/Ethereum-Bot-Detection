from src.Datamodels.Table import Table
import pandas as pd

class Tab(Table):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "data", ["large"]
        super().__init__(name, chapter, runs)

    def set_outnames(self):
        self.outnames = [self.name]

    def create_tex_code(self):
        df = self.load_data()

        rename = {
            "N Self Transactions": "Number of Self Transactions",
            "N Transactions": "Number of Transactions",
            "Timeframe traded in": "Timeframe of Activity",
            "Max same-block Out-Transactions": "Max. Same-Block Out-Transactions",
            "Max same-block In-Transactions": "Max. Same-Block In-Transactions",
        }


        df.columns = ["Statistics", "Value"]
        # make the word "block" uppercase if the value is a string
        f = lambda x: x.replace("block", "Block") if isinstance(x, str) else x
        df["Value"] = df["Value"].apply(f)
        # rename elements in Statistics column if they are in rename
        df["Statistics"] = df["Statistics"].apply(lambda x: rename[x] if x in rename else x)
        df = df.set_index("Statistics")
        df.index.name = "Statistics of " + df["Value"]["Address"][:10] + "..."
        # drop row "Address"
        df = df.drop("Address", axis=0)
        tex_string = df.to_latex(index=True)
        return tex_string


if __name__ == "__main__":
    prefix = "../.."
    v = Tab()
    v.prefix = prefix
    v.create_and_save()