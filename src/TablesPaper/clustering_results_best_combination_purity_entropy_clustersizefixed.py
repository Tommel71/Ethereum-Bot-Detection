from src.Datamodels.Table import Table
import pandas as pd

class Tab(Table):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)


    def set_outfolder(self):
        self.outfolder = f"{self.prefix}/paper_assets/tables"

    def create_tex_code(self):
        df = self.load_data()


        rename_columns = {
            "purity": "Purity",
            "entropy": "Entropy",
            "size": "Size",
        }
        df = df.rename(columns=rename_columns)
        df.index.name = "Cluster"

        df = df[df.Size >= 15]

        additional_data_path = f"{self.prefix}/outputs/large/tables_pickled/results/clustering_results_best_combination_category_matrix_clustersizefixed.pkl"
        additional_data = pd.read_pickle(additional_data_path)


        # get max column for each row
        max_column = additional_data.idxmax(axis=1)
        # create mapping
        cluster_id_to_category = max_column.to_dict()

        # create majority column
        df["Majority"] = df.index.map(cluster_id_to_category)

        tex_ = df.to_latex(index=True, escape=False)
        return tex_


if __name__ == "__main__":
    prefix = "../.."
    v = Tab()
    v.prefix = prefix
    v.create_and_save()