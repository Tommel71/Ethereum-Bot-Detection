from src.Datamodels.Visualisation import Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

class Vis(Visualisation):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)
        self.scaling_factor = 1.5

    def create_visualisation(self):
        category_matrix_normalised = self.load_data()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(category_matrix_normalised, annot=True, ax=ax, cmap="Blues")
        plt.tight_layout()
        #title = "Confusion Matrix Multiclass MEV"
        x_label = "Predicted"
        y_label = "True"

        #ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)



if __name__ == "__main__":
    prefix = "../.."
    v = Vis()
    v.prefix = prefix
    v.render_visualisation()