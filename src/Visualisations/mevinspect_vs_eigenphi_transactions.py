from src.Datamodels.Visualisation import Visualisation
import matplotlib.pyplot as plt
import pandas as pd
import datetime

class Vis(Visualisation):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "background", ["large"]
        super().__init__(name, chapter, runs)

    def create_visualisation(self):
        labels, mev_counts, eig_counts, common_counts = self.load_data()
        plt.figure(figsize=(5, 3))
        bar_width = 0.2
        index = range(len(labels))

        plt.bar(index, mev_counts, bar_width, label='MEV-Inspect', align='center')
        plt.bar([i + bar_width for i in index], eig_counts, bar_width, label='Eigenphi', align='center')
        plt.bar([i + 2 * bar_width for i in index], common_counts, bar_width, label='Common', align='center')

        plt.xlabel('MEV-Type')
        plt.ylabel('\# Transactions')
        #plt.title('Transactions found')
        plt.xticks([i + bar_width for i in index], labels)
        plt.legend()


if __name__ == "__main__":
    v = Vis()
    prefix = "../.."
    v.prefix = prefix
    v.render_visualisation()