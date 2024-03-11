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
        df.index.name = "Fine-Grained Reconciled"

        tex_string = df.to_latex(index=True)
        return tex_string


if __name__ == "__main__":
    prefix = "../.."
    v = Tab()
    v.prefix = prefix
    v.create_and_save()