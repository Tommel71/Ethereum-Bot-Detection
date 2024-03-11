from src.Datamodels.Table import Table
import pandas as pd

class Tab(Table):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "data", ["large"]
        super().__init__(name, chapter, runs)


    def create_tex_code(self):
        df = self.load_data()

        # change order of index to ["Bot", "Unclear", "Human"]
        df = df.reindex(["Bot", "Unclear", "Human"])

        tex_string = df.to_latex(index=True)
        return tex_string


if __name__ == "__main__":
    prefix = "../.."
    v = Tab()
    v.prefix = prefix
    v.create_and_save()