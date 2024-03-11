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
        plt.figure(figsize=(5, 5))
        df["time"].hist(bins=24, range=[0, 24])
        plt.xlabel("Time of the Day [hours]")
        plt.ylabel("Number of Transactions")
        #plt.title("Distribution of Transactions")
