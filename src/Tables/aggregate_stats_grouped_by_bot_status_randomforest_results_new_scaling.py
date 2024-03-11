from src.Datamodels.Table import Table
import pandas as pd
from tools import get_window_names_mapping
from src.Tables.aggregate_stats_grouped_by_bot_status_randomforest_results_keep_scaling import Tab as T2

class Tab(Table):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def create_tex_code(self):
        return T2.create_tex_code(self)


if __name__ == "__main__":
    prefix = "../.."
    v = Tab()
    v.prefix = prefix
    v.create_and_save()