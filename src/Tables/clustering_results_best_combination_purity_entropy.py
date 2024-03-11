from src.Datamodels.Table import Table
import pandas as pd

class Tab(Table):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def set_outnames(self):
        self.outnames = [self.name + "1", self.name + "2"]

    def create_tex_code(self):
        df = self.load_data()


        rename_columns = {
            "purity": "Purity",
            "entropy": "Entropy",
            "size": "Size",
        }
        df = df.rename(columns=rename_columns)
        df.index.name = "Cluster"


        length = len(df)
        # split df into two
        df1 = df.iloc[:int(length+1)//2]
        df2 = df.iloc[int(length+1)//2:]

        # if df2 is shorter than df1, add one row of empty strings
        if len(df2) < len(df1):
            df2 = pd.concat([df2, pd.DataFrame([[""]*len(df2.columns)], columns=df2.columns)])
            df2.index = list(df2.index[:-1]) + [" "]
            df2.index.name = "Cluster"

        # create tex code
        tex_string1 = df1.to_latex(index=True)
        tex_string2 = df2.to_latex(index=True)



        return [tex_string1, tex_string2]


if __name__ == "__main__":
    prefix = "../.."
    v = Tab()
    v.prefix = prefix
    v.create_and_save()