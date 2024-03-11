# model wrappers implementing train and predict methods
from abc import abstractmethod
import pandas as pd
import pickle
from tools import flatten

class Model:

    @abstractmethod
    def predict(self, x):
        pass

    def save(self, path):
        # use pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

class FromDataModel(Model):

    def __init__(self, prefix="."):
        self.txHashes = None
        self.prefix = prefix


    def predict(self, txHash):
        return txHash in self.txHashes


class LiquidationEigenphi(FromDataModel):

    def __init__(self, prefix="."):
        super().__init__(prefix)
        self.txHashes = set(pd.read_csv(f"{prefix}/data/eigenphi/liquidation_all.csv")["transactionHash"])

class SandwichEigenphi(FromDataModel):

    def __init__(self, prefix="."):
        super().__init__(prefix)
        txes = list(pd.read_csv(f"{prefix}/data/eigenphi/sandwich_all.csv")["attackerTxs"])
        list_of_lists = [tx.split(" ") for tx in txes]
        self.txHashes = set(flatten(list_of_lists))

class ArbitrageEigenphi(FromDataModel):

    def __init__(self, prefix="."):
        super().__init__(prefix)
        self.txHashes = set(pd.read_csv(f"{prefix}/data/eigenphi/arbitrage_all.csv")[" txHash"])


class LiquidationMevInspect(FromDataModel):

    def __init__(self, prefix="."):
        super().__init__(prefix)
        self.txHashes = set(pd.read_csv(f"{prefix}/data/mev_inspect_predictions/liquidations.csv")["tx_hash"])

class SandwichMevInspect(FromDataModel):

    def __init__(self, prefix="."):
        super().__init__(prefix)
        self.txHashes = set(pd.read_csv(f"{prefix}/data/mev_inspect_predictions/sandwiches.csv")["tx_hash"])


class ArbitrageMevInspect(FromDataModel):

    def __init__(self, prefix="."):
        super().__init__(prefix)
        self.txHashes = set(pd.read_csv(f"{prefix}/data/mev_inspect_predictions/arbitrages.csv")["tx_hash"])



class MultiModel(Model):

        def __init__(self, models: list):
            self.models = models

        def predict(self, x):
            return any([m.predict(x) for m in self.models])
