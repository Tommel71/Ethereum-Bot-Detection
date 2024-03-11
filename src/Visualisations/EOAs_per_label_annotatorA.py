from src.Datamodels.Visualisation import Visualisation
import matplotlib.pyplot as plt

class Vis(Visualisation):

    def __init__(self):
        script_name = __file__
        name, chapter, runs = script_name, "data", ["large"]
        super().__init__(name, chapter, runs)

    def create_visualisation(self):
        df_agg_A = self.load_data()
        plt.figure(figsize=(5, 5))
        plt.locator_params(nbins=5)
        #plt.grid(axis="y")
        plt.grid(b=None)

        # Plotting horizontal bar chart
        # color the largest bar blue and the others orange

        # get first color of standard color palette
        palette = plt.get_cmap('tab10').colors
        # add fourth channel to all colors
        palette = [list(color) + [1] for color in palette]

        colors = [palette[1]] * (len(df_agg_A)-1)
        colors += [palette[0]]


        plt.barh(df_agg_A.index[::-1], df_agg_A.values[::-1], color=colors)
        # Adjusting x-axis limits to ensure there's space for text
        plt.xlim(0, max(df_agg_A.values) * 1.15)

        for i, v in enumerate(df_agg_A.values[::-1]):
            # Dynamic text positioning
            offset = 3 if v < max(df_agg_A.values) * 0.8 else 0.5
            plt.text(v + offset, i - 0.3, str(v), color='black', fontweight='bold')

        plt.legend(["Bot Subcategory"], loc="lower right")
        #plt.title("Number of EOAs per Label")
        plt.xlabel("Number of EOAs")
        plt.ylabel("Label")

if __name__ == "__main__":
    prefix = "../.."
    v = Vis()
    v.prefix = prefix
    v.render_visualisation()