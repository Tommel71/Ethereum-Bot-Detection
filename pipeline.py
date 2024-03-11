from tools import load_configs, set_configs, load_plotting_settings_static
from src.Labeling import annotation_single, process_labeling
from src.Preprocessing.populate_postgres import Preprocess
from src.Aggregate import Aggregate
from scripts.mevinspect_vs_eigenphi import run as mevinspect_vs_eigenphi
from scripts.timeframes_of_interest import run as timeframes_of_interest
from scripts.handle_compressed_csvs import unpack_all, delete_all_unpacked
from src.Datamodels.PipelineComponent import PipelineComponent
from src.Investigate import Analysis
import pandas as pd
from tools import save_data_for_figure, log_time
from scripts.improve_base_data import create_relative_transaction_index
from scripts.get_price_data import get_all_price_data
from tqdm import tqdm
from scripts import render_powerpoints, render_visualisations, render_tables
import update_latex
from tools import save_table
import os
import shutil

# The main sample is the 100k sample that has the 2 test blocks at the end
# Most of the analysis is done on this.
main_sample = "large"
settingnames = ["largesample1", "largesample2", "largesample3", "largesample4", main_sample]

prediction_files = ["randomforest_results_keep_scaling", "randomforest_results_new_scaling"]
class FinalPipeline(PipelineComponent):

    @log_time
    def onetime_setup(self):
        get_all_price_data(self.prefix)

    @log_time
    def populate_db(self):

        for name in settingnames:

            self.logger.debug(f"Start DB population for {name}")

            set_configs(name, self.prefix)
            configs = load_configs(self.prefix) # reload configs to get the correct settings, never use self.config in this class
            #load_plotting_settings_static()

            ### unpack csvs
            unpack_all(self.prefix)
            create_relative_transaction_index(self.prefix) # ugly hack. otherwise would need more research into how to make it faster in SQL

            # create the DB if it does not exist
            ppc = PipelineComponent(configs, self.prefix)
            ppc.create_postgres_database()
            ppc.empty_database()

            ### Preprocessing and population of the database
            p = Preprocess(configs, self.prefix)
            p.run()

            ### delete csvs to release storage
            delete_all_unpacked(self.prefix)

    def delete_cache(self):
        """
        Some results are cached because they take a long time to compute or query. For a new setup, changes in features
        or settings, price data etc. this cache must be deleted.
        :return:
        """

        for name in settingnames:
            set_configs(name, self.prefix)
            configs = load_configs(self.prefix)
            cache_folder = configs["General"]["PREFIX_DB"] + "/cache"
            # folder is not empty
            if os.path.exists(cache_folder):
                shutil.rmtree(cache_folder)


    @log_time
    def ETL(self):

        for name in settingnames:

            self.logger.debug(f"Start ETL for {name}")
            set_configs(name, self.prefix)
            configs = load_configs(self.prefix) # reload configs to get the correct settings, never use self.config in this class
            ### Aggregation
            a = Aggregate(configs, self.prefix)
            a.ETL()

            # log
            self.logger.debug(f"Finished ETL for {name}")


        print("pipeline finished")

    @log_time
    def train(self):
        """
        Train and save model
        :return:
        """

        set_configs("large", self.prefix)
        configs = load_configs(self.prefix)
        an = Analysis(configs, self.prefix)
        an.train()

    @log_time
    def calculate_eval(self):
        """
        Load trained model and apply it to all the samples, save results
        :return:
        """
        for name in settingnames:
            set_configs(name, self.prefix)
            configs = load_configs(self.prefix)
            an = Analysis(configs, self.prefix)
            an.evaluate()

    @log_time
    def combine_aggregate_stats_grouped_by_bot_status(self):

        for prediction_file in prediction_files:

            data = []
            for name in settingnames:
                set_configs(name, self.prefix)
                configs = load_configs(self.prefix)
                an = Analysis(configs, self.prefix)
                statistics = an.calculate_aggregate_stats_grouped_by_bot_status(prediction_file)
                print(statistics)
                data.append(statistics)

            df = pd.DataFrame(data, index=settingnames)
            filename, chapter = f"aggregate_stats_grouped_by_bot_status_{prediction_file}", "results"
            save_table(df, filename, chapter, self.prefix)

    @log_time
    def calculate_feature_stats(self):

        for name in settingnames:
            set_configs(name, self.prefix)
            configs = load_configs(self.prefix)
            an = Analysis(configs, self.prefix)
            an.calculate_feature_stats()


    @log_time
    def investigate_feature_drift(self):

        data = []
        for name in settingnames:
            set_configs(name, self.prefix)
            configs = load_configs(self.prefix)
            an = Analysis(configs, self.prefix)
            feature_means, feature_stds = an.load_feature_stats()
            data.append((feature_means, feature_stds))

        means_df = pd.concat([x[0] for x in data], axis=1, keys=settingnames)
        stds_df = pd.concat([x[1] for x in data], axis=1, keys=settingnames)
        means_df.columns = means_df.columns.droplevel(1)

        #path = "E:\Masterthesis\largesample4\cache/" + "features.pkl"
        #aa = pd.read_pickle(path)
        means_df.fillna(0, inplace=True)
        stds_df.columns = stds_df.columns.droplevel(1)

        # take the large sample as reference
        reference = means_df[[main_sample]]
        reference_std = stds_df[[main_sample]]

        # from each col subtract reference
        ddd = means_df.subtract(reference.values, axis=0)
        ddd_norm = ddd.divide(reference_std.values, axis=0)

        # settingnames without main_sample
        settingnames_small = [x for x in settingnames if x != main_sample]

        data = ddd_norm, settingnames_small

        filename, chapter, prefix = "feature_drift_heatmap", "results", self.prefix
        save_data_for_figure(data, filename, chapter, prefix)


    @log_time
    def calculate_DB_statistics(self):
        list_of_acc_dfs = []
        for name in settingnames:
            set_configs(name, self.prefix)
            configs = load_configs(self.prefix)
            an = Analysis(configs, self.prefix)
            an.calculate_DB_statistics()

            accs =an.get_accs()
            list_of_acc_dfs.append(accs)

        # set main
        set_configs("large", self.prefix)
        configs = load_configs(self.prefix)
        an = Analysis(configs, self.prefix)
        an.calculate_acc_statistics(list_of_acc_dfs)

    def investigate_DB_statistics(self):
        data = []
        for name in settingnames:
            set_configs(name, self.prefix)
            configs = load_configs(self.prefix)
            an = Analysis(configs, self.prefix)
            statistics = an.load_DB_statistics()
            data.append(statistics)

        set_configs("large", self.prefix)
        configs = load_configs(self.prefix)
        an = Analysis(configs, self.prefix)
        accs_deduplicated = an.load_acc_statistics()
        keys = configs["General"]["window_names_internal"] # list
        targets = configs["General"]["window_names"] # list
        df = pd.concat(data)
        df.index = settingnames


        # map index to targets and order the way of targets
        map = dict(zip(keys, targets))
        df.index = df.index.map(map)
        # sort index according to targets
        df = df.loc[targets]


        # add total line and style
        df.loc["Total"] = df.sum()

        #replace totals of EOAs and CAs with actual totals (deduplicated)
        df.loc["Total", "EOA"] = accs_deduplicated["EOA"]["address"]
        df.loc["Total", "CA"] = accs_deduplicated["CA"]["address"]

        filename, chapter = f"DB_statistics", "results"

        save_table(df, filename, chapter, self.prefix)

    @log_time
    def analysis(self):
        ### Labeling
        annotation_single.inspect_specific(self.prefix)
        process_labeling.run(self.prefix)

        ### analysis
        run = "large"
        set_configs(run, self.prefix)
        timeframes_of_interest(self.prefix)
        mevinspect_vs_eigenphi(self.prefix)

        #for prediction_file in prediction_files:
        #    df = pd.read_csv(self.prefix + f"/{run}/predictions/{prediction_file}.csv", index_col=0)
        #    filename, chapter, prefix = f"meta_percentages_{prediction_file}", "results", self.prefix
        #    save_data_for_figure(df, filename, chapter, prefix)

    @log_time
    def investigate_value(self):

        for prediction_file in prediction_files:

            data = {}
            data_tokens = {}
            for name in tqdm(settingnames):
                set_configs(name, self.prefix)
                configs = load_configs(self.prefix)
                an = Analysis(configs, self.prefix)
                cumulative_df, value_df_tokens_bot, value_df_tokens_non_bot = an.get_cumulative_value(prediction_file)
                data[name] = cumulative_df
                data_tokens[name] = (value_df_tokens_bot, value_df_tokens_non_bot)

            save_data_for_figure(data, f"cumulative_value_{prediction_file}", "results", self.prefix)
            save_data_for_figure(data_tokens, f"cumulative_value_tokens_{prediction_file}", "results", self.prefix)


    @log_time
    def run_experiments(self):

        run = "large"
        set_configs(run, self.prefix)
        configs = load_configs(self.prefix)
        an = Analysis(configs, self.prefix)
        an.run_experiments()

    def render(self):

        render_powerpoints.render_powerpoints(self.prefix)
        render_visualisations.run(self.prefix)
        render_tables.run(self.prefix)
        update_latex.run(self.prefix)

if __name__ == "__main__":
    prefix = "."
    configs_1 = load_configs(prefix)
    finalpipeline = FinalPipeline(configs_1, prefix)
    finalpipeline.onetime_setup()
    finalpipeline.delete_cache()
    finalpipeline.populate_db()
    finalpipeline.ETL()
    # analysis on the main observation window
    finalpipeline.run_experiments()
    # longitudinal study
    finalpipeline.train()
    finalpipeline.calculate_eval() # takes like 20 minutes, when stuff is already cached
    finalpipeline.combine_aggregate_stats_grouped_by_bot_status()
    finalpipeline.calculate_feature_stats()
    finalpipeline.investigate_feature_drift()
    finalpipeline.investigate_value()
    finalpipeline.analysis()
    finalpipeline.calculate_DB_statistics()
    finalpipeline.investigate_DB_statistics()
    finalpipeline.render()