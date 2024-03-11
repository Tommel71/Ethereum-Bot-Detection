from src.Datamodels.Visualisation import Visualisation
import matplotlib.pyplot as plt

class Vis(Visualisation):

    def __init__(self):
        script_name = __file__
        name, chapter, runs = script_name, "data", ["large"]
        super().__init__(name, chapter, runs)

    def create_visualisation(self):
        df_agg_B = self.load_data()
        plt.figure(figsize=(5, 5))
        plt.locator_params(nbins=5)
        #plt.grid(axis="y")
        # remove grid
        plt.grid(b=None)
        # add numbers next to bars
        palette = plt.get_cmap('tab10').colors
        # add fourth channel to all colors
        palette = [list(color) + [1] for color in palette]
        colors = [palette[1]] * (len(df_agg_B)-1)
        colors += [palette[0]]


        plt.barh(df_agg_B.index[::-1], df_agg_B.values[::-1], color=colors)

        # Adjusting x-axis limits to ensure there's space for text
        plt.xlim(0, max(df_agg_B.values) * 1.15)

        for i, v in enumerate(df_agg_B.values[::-1]):
            # Dynamic text positioning
            offset = 3 if v < max(df_agg_B.values) * 0.8 else 0.5
            plt.text(v + offset, i - 0.22, str(v), color='black', fontweight='bold')

        #plt.title("Number of EOAs per Label")
        # add legend element that explains that orange means bot subcategory

        plt.legend(["Bot Subcategory"], loc="lower right")

        plt.xlabel("Number of EOAs")
        plt.ylabel("Label")

if __name__ == "__main__":
    prefix = "../.."
    v = Vis()
    v.prefix = prefix
    v.render_visualisation()