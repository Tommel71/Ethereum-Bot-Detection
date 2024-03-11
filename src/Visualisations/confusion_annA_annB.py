from src.Datamodels.Visualisation import Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

class Vis(Visualisation):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "data", ["large"]
        super().__init__(name, chapter, runs)

    def create_visualisation(self):
        confusion_df = self.load_data()
        plt.figure(figsize=(10, 9))
        # add label to the color bar
        sns.heatmap(confusion_df, annot=True, cmap="Blues", fmt='g', cbar_kws={'label': 'Number of EOAs'})
        #plt.title("Confusion Matrix Disagreeing Labels")
        plt.xlabel("Annotator B")
        plt.ylabel("Annotator A")
        plt.tight_layout()


if __name__ == "__main__":
    prefix = "../.."
    v = Vis()
    v.prefix = prefix
    v.render_visualisation()