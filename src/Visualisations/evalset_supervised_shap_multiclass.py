import pandas as pd
import shap
from src.Datamodels.Visualisation import Visualisation
import matplotlib.pyplot as plt
from tools import feature_name_mapper
import numpy as np

class Vis(Visualisation):

    def __init__(self):
        # get name of this script
        script_name = __file__
        name, chapter, runs = script_name, "results", ["large"]
        super().__init__(name, chapter, runs)

    def create_visualisation(self):
        X, y, pipeline, columns, y_num = self.load_data()
        columns = feature_name_mapper(columns)

        renaming = {
            "Tx-Time-OutHourlyEntropy": "Out-TX-Entropy",
            "Tx-Block-NTxPerBlock": "Out-TX-Per-Block",
            "Tx-Time-OutTransactionFrequency": "Out-TX-Frequency",
            "Tx-GasPriceMax": "TX-GasPrice-Max",
            "Tx-Time-OutSleepiness": "Out-TX-Sleepiness",
            "Tx-GasQuantile95": "TX-Gas-Quantile95",
            "Scb-Eb-Transfer-TransfersPerBlock": "Token-Event-Transfer-Per-Block",
            "Tx-Time-OutQuantile95": "Out-TX-Quantile95",
            "Tx-GasMean": "TX-Gas-Mean",
            "Scb-Eb-Transfer-ValueMedian": "Token-Event-Value-Median",
            "Tx-GasMin": "TX-Gas-Min",
            "Tx-GasMax": "TX-Gas-Max",
            "Tx-GasMedian": "TX-Gas-Median",
        }
        y = pd.DataFrame(y)
        y["num"] = y_num
        mapping = y.drop_duplicates().reset_index(drop=True)["type"].to_dict()
        columns = [renaming.get(x, x) for x in columns]
        columns = [renaming.get(x, x) for x in columns]

        np.random.seed(0)
        model = pipeline.steps[-1][1]
        explainer = shap.TreeExplainer(model)
        X_scaled = pipeline.steps[0][1].fit_transform(X)
        X_imputed = pipeline.steps[1][1].fit_transform(X_scaled)
        shap_values = explainer.shap_values(X_imputed)

        shap.summary_plot(shap_values, X_imputed, feature_names=columns, plot_type="bar", class_names=[mapping[i] for i in model.classes_], max_display=5, show=False)
        plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)")

if __name__ == "__main__":
    prefix = "../.."
    v = Vis()
    v.prefix = prefix
    v.render_visualisation()