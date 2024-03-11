from tools import load_configs_by_name
import os
import pandas as pd
from tools import standard_latex_formatting

class Table:

    def __init__(self, name, chapter, runs, prefix="."):
        self.runs = runs
        self.prefix = prefix
        self.chapter = chapter
        self.name = name
        # clean name
        self.name = self.name.split("\\")[-1].split(".")[0]
        self.scaling_factor = 1
        self.outnames = [name] # usually there is just one output

    def set_outnames(self):
        self.outnames = [self.name]

    def set_outfolder(self):
        self.outfolder = f"{self.prefix}/outputs/{self.configs['General']['run_name']}/latex_snippets/{self.chapter}"

    def set_config(self, config_name):
        self.configs = load_configs_by_name(config_name, self.prefix)
        self.infolder = f"{self.prefix}/outputs/{self.configs['General']['run_name']}/tables_pickled/{self.chapter}"
        self.set_outfolder()

    def set_settings(self):
        pass

    def create_tex_code(self):
        pass

    def save_as_tex(self, tex_strings):
        if not os.path.exists(self.outfolder):
            os.makedirs(self.outfolder)

        for outname, tex_string in zip(self.outnames, tex_strings):
            tex_string = standard_latex_formatting(tex_string)
            with open(f"{self.outfolder}/{outname}.tex", "w", encoding="utf-8") as f:
                f.write(tex_string)

    def load_data(self):
        file = f"{self.infolder}/{self.name}.pkl"
        df = pd.read_pickle(file)
        df = df.round(3)

        return df

    def create_and_save(self):
        self.set_outnames()
        for run in self.runs:
            self.set_config(run)
            self.set_settings()
            tex_strings = self.create_tex_code()
            # if tex_strings is not a list, make it a list
            if not isinstance(tex_strings, list):
                tex_strings = [tex_strings]

            self.save_as_tex(tex_strings)
