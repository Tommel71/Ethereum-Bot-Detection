from src.Datamodels.Table import Table
import pandas as pd
from scripts.get_n_clusters import get_n_clusters_elbow

class Tab(Table):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def set_outnames(self):
        self.outnames = [self.name]

    def set_outfolder(self):
        self.outfolder = f"{self.prefix}/paper_assets/tables"

    def create_tex_code(self):
        df = self.load_data()
        # in the multiindex replace "gmm" with "GMM"
        df.index = df.index.set_levels(df.index.levels[0].str.replace("gmm", "GMM"), level=0)


        n_clusters = get_n_clusters_elbow(self.prefix)
        df.index.names = ["Algorithm", "Imputation", "thead[[[Dimension. latexnewline Reduction]]]"]
        df = df[['Purity', 'Entropy']]

        # add n_clusters
        df["Clusters"] = n_clusters

        tex_string = df.to_latex(index=True)
        return tex_string


if __name__ == "__main__":
    prefix = "../.."
    v = Tab()
    v.prefix = prefix
    v.create_and_save()