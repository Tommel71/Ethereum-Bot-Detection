from tools import load_configs_by_name
from matplotlib import pyplot as plt
import seaborn as sns
import os
import pickle

class Visualisation:

    def __init__(self, name, chapter, runs, prefix="."):
        self.runs = runs
        self.prefix = prefix
        self.chapter = chapter
        self.name = name
        # clean name
        self.name = self.name.split("\\")[-1].split(".")[0]
        self.scaling_factor = 1


    def set_outfolder(self):
        self.outfolder = f"{self.prefix}/outputs/{self.configs['General']['run_name']}/figures/{self.chapter}"

    def set_config(self, config_name):
        self.configs = load_configs_by_name(config_name, self.prefix)
        self.infolder = f"{self.prefix}/outputs/{self.configs['General']['run_name']}/figtables_pickled/{self.chapter}"
        self.set_outfolder()

    def set_settings(self):
        factor = self.scaling_factor
        sns.set_style('whitegrid')
        plt.rcParams.update({
            "text.usetex": True,
            'font.family': 'serif',
            'font.serif': 'Computer Modern Roman',
            "xtick.bottom": True,
            "ytick.left": True,
            'axes.labelsize': factor * self.configs["Plotting"]["label_size"],
            'font.size': factor * self.configs["Plotting"]["font_size"],
            'legend.fontsize': factor * 12,
            'xtick.labelsize': factor * self.configs["Plotting"]["tick_size"],
            'ytick.labelsize': factor * self.configs["Plotting"]["tick_size"],
            'figure.titlesize': factor * self.configs["Plotting"]["title_size"],
            'axes.titlesize': factor * self.configs["Plotting"]["title_size"],
            'figure.dpi': 96,
        })

    def create_visualisation(self):
        pass

    def save_visualisation(self):
        if not os.path.exists(self.outfolder):
            os.makedirs(self.outfolder)

        plt.savefig(f"{self.outfolder}/{self.name}.pdf", dpi=600, bbox_inches='tight')
        plt.close()

    def load_data(self):
        file = f"{self.infolder}/{self.name}.pkl"
        with open(file, "rb") as f:
            data = pickle.load(f)

        return data

    def render_visualisation(self):
        for run in self.runs:
            self.set_config(run)
            self.set_settings()
            self.create_visualisation()
            self.save_visualisation()