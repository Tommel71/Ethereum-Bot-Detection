from src.Datamodels.Table import Table
import pandas as pd

class Tab(Table):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def set_outnames(self):
        self.outnames = [self.name]

    def create_tex_code(self):
        df = self.load_data()
        #

        # reorder so Arbitrage, Sandwich and Liquidation is first
        df = df.reindex(["Arbitrage", "Sandwich", "Liquidation"] + list(df.index.drop(["Arbitrage", "Sandwich", "Liquidation"])))
        # add "Binary " to the first level of the columns
        df.columns.set_levels(["Binary " + x for x in df.columns.levels[0]], level=0, inplace=True)
        # call the first column level Dataset and the second Metric
        #df.columns.names = ["Dataset", "Metric"]



        tex_string = df.to_latex(index=True,multicolumn_format='c|', multicolumn=True)
        # change last c| to c because there is no column after the last multicolumn
        tex_string = tex_string[::-1].replace("|c", "c", 1)[::-1]
        # get the left and right index of the xyz part of \begin{tabular}{xyz}
        # use regex
        import re
        regex = r"\\begin{tabular}{(.*)}"
        match = re.search(regex, tex_string)
        left_index_ = match.span()[0]
        left_index = left_index_ + len("\\begin{tabular}{")
        right_index_ = match.span()[1]
        right_index = right_index_ - len("}")
        colstring = tex_string[left_index:right_index]
        # add vertical lines after every second column
        rowindexsize = 1
        level1_colsize = 2
        rowsnippet = colstring[0:rowindexsize] + "|"
        improved_colstring = rowsnippet +  "|".join([colstring[level1_colsize*i + rowindexsize:level1_colsize*(i+1) + rowindexsize]
                              for i in range(0, int(len(colstring)/level1_colsize))])

        # insert colstring
        latex_with_vertical_lines = tex_string[0:left_index] + improved_colstring + tex_string[right_index:]
        return latex_with_vertical_lines


if __name__ == "__main__":
    prefix = "../.."
    v = Tab()
    v.prefix = prefix
    v.create_and_save()