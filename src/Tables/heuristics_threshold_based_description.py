from src.Datamodels.Table import Table
import pandas as pd

class Tab(Table):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def create_tex_code(self):
        df = self.load_data()

        renaming = {
            "tx__custom__n_tx_per_block": "Out-TX-Per-Block",
            "tx__time__intime_sleepiness": "In-TX-Sleepiness",
            "tx__time__outtime_sleepiness": "Out-TX-Sleepiness",
            "tx__time__intime_transaction_frequency": "In-TX-Frequency",
            "tx__time__outtime_transaction_frequency": "Out-TX-Frequency",
            "tx__value__tvc": "TX-Value-TVC",
            "scb__eb__transfer__value__tvc": "SCB-Value-TVC", # tk stands for token
        }


        # ugly and loops but df is small so it's ok
        # replace all strings in the Description column using the renaming dict
        # iterate over all cells
        for i in range(len(df)):
            for j in range(len(df.columns)):
                # if the cell contains a string
                if isinstance(df.iloc[i, j], str):
                    # replace the string
                    for key, value in renaming.items():
                        df.iloc[i, j] = df.iloc[i, j].replace(key, value)

        with pd.option_context("max_colwidth", 1000):

            # create tex code, dont dot dot dot
            tex_string = df.to_latex(index=False)


        return tex_string


if __name__ == "__main__":
    prefix = "../.."
    v = Tab()
    v.prefix = prefix
    v.create_and_save()