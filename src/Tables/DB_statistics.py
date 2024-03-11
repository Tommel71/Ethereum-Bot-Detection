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




        # write in terms of millions and format it as M
        df = df / 1000000
        df = df.round(2)
        df = df.astype(str) + "M"

        df.index.name = "Observation Window"

        def df_style(val):
            return "textbf:--rwrap;"


        last_row = pd.IndexSlice[df.index[df.index == "Total"], :]

        styler = df.style.applymap(df_style, subset=last_row)
        # format only the entry in the index that contains "Total"
        f = lambda x: r"\textbf{" + x + r"}" if "Total" in x else x
        styler = styler.format_index(f, axis=0, escape="latex")
        tex_string = styler.to_latex(hrules=True)

        return tex_string


if __name__ == "__main__":
    prefix = "../.."
    v = Tab()
    v.prefix = prefix
    v.create_and_save()