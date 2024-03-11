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
        df = self.load_data()
        df["functionName"] = df["functionName"].apply(lambda x: "Transfer" if x == "" else x)
        # remove arguments
        df["functionName"] = df["functionName"].apply(lambda x: x.split("(")[0])
        df["functionNameShort"] = df["functionName"].apply(lambda x: x[:10])
        # replace "" with "Transfer"
        vc = df["functionNameShort"].value_counts()[:10]

        plt.figure(figsize=(5, 5))
        plt.locator_params(nbins=5)
        #plt.grid(axis="y")
        plt.barh(vc.index[::-1], vc.values[::-1])

        plt.xlabel("Number of Transactions")
        plt.ylabel("Function Name (truncated)")
        #plt.title("Most used Functions")
