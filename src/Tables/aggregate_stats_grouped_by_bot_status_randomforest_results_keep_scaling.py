from src.Datamodels.Table import Table
import pandas as pd
from tools import get_window_names_mapping

class Tab(Table):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def create_tex_code(self):
        df = self.load_data()

        window_names_mapping = get_window_names_mapping(self.configs)

        rename_col_mapping = {
            "n_bot_tx": "thead[[[N TX Sent latexnewline Bots]]]",
            "percentage_bot_tx": "thead[[[Pct. TX Sent latexnewline Bots]]]",
            "n_bots": "N Bots",
        #    "n_addresses": "N EOAs",
            "percentage_bots": "Pct. Bots",
            "average_sci_bot": "thead[[[Average latexnewline SCI Bots]]]",
            "average_sci_non_bot": "thead[[[Average latexnewline SCI Humans]]]",
        }

        drop_cols = ["n_addresses"]

        format_as_M = ["n_bot_tx", "n_bots"]
        percentage_cols = ["percentage_bot_tx", "percentage_bots", "average_sci_bot", "average_sci_non_bot"]

        df = df.drop(columns=drop_cols)

        for col in format_as_M:
            df[col] = df[col].apply(lambda x: f"{x / 1e6:.1f}M")

        for col in percentage_cols:
            df[col] = df[col].apply(lambda x: f"{100*x:.2f}%")

        # use the order of the window names as defined in the config file
        df = df.reindex(window_names_mapping.keys())

        df = df.rename(columns=rename_col_mapping)
        df = df.rename(index=window_names_mapping)

        df.index.name = "Observation Window"
        tex_string = df.to_latex(index=True)
        return tex_string


if __name__ == "__main__":
    prefix = "../.."
    v = Tab()
    v.prefix = prefix
    v.create_and_save()