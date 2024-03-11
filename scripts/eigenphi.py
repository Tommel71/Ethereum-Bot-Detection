"""
Quick script to see how usable the MEV data is
"""


from tools import load_block
from src.Datamodels.Models import ArbitrageEigenphi, LiquidationEigenphi, SandwichEigenphi, MultiModel
from src.Datamodels.Models import LiquidationMevInspect, ArbitrageMevInspect, SandwichMevInspect
import pandas as pd
from tqdm import tqdm
prefix = ".."

models_eigenphi = [
    ArbitrageEigenphi(prefix),
    LiquidationEigenphi(prefix),
    SandwichEigenphi(prefix)
]
model_eigenphi = MultiModel(models_eigenphi)

models_mevinspect = [
    ArbitrageMevInspect(prefix),
    LiquidationMevInspect(prefix),
    SandwichMevInspect(prefix)

]
model_mevinspect = MultiModel(models_mevinspect)

data = []
n_blocks = 1000
start_block = 15000000
for i in tqdm(range(start_block, start_block + n_blocks)):

    path = f"{prefix}/data/mev_inspect_blocks/{i}.json"
    block = load_block(path)
    data_i = []
    for trace in block.traces:
        if not trace.trace_address:
            tx_hash = trace.transaction_hash
            from_address = trace.action["from"]
            data_i.append((tx_hash, from_address, trace.block_number))

    data += data_i


def count_score(timeseries):
    return len(timeseries) > n_blocks/4


def speed_score(timeseries):
    # if any block is hit more than twice return True
    return any(timeseries.value_counts() > 2)


df = pd.DataFrame(data, columns=["txHash", "fromAddress", "blockNumber"])
df["isMev_eigenphi"] = df["txHash"].apply(model_eigenphi.predict)
df["isMev_mevinspect"] = df["txHash"].apply(model_mevinspect.predict)
df["isMev"] = df["isMev_mevinspect"] | df["isMev_eigenphi"]

mev_count_tx_eigenphi = df.groupby("isMev_eigenphi").count()
mev_count_tx_mevinspect = df.groupby("isMev_mevinspect").count()
mev_count_tx = df.groupby("isMev").count()

wallets = df.drop_duplicates(subset=["fromAddress"])
mev_per_wallet = pd.DataFrame((df.groupby(["txHash", "fromAddress"]).sum()["isMev"] > 0).groupby("fromAddress").sum() > 0)
mev_count_wallet = mev_per_wallet.reset_index().groupby("isMev").count()
print(mev_count_tx)
print(mev_count_wallet)
print("Percentage MEV tx:", mev_count_tx["txHash"][True] / mev_count_tx["txHash"].sum())
print("Percentage MEV wallet:", mev_count_wallet.loc[True]["fromAddress"] / mev_count_wallet.sum())
count = df[["fromAddress","blockNumber"]].groupby("fromAddress").agg(count_score)
speed = df[["fromAddress","blockNumber"]].groupby("fromAddress").agg(speed_score)
print("Wallets that get caught by the count heuristic:", count.sum()["blockNumber"]/len(count))
print("Wallets that get caught by the speed heuristic:", speed.sum()["blockNumber"]/len(speed))

print("Wallets that have 1 marker", pd.concat([count, speed, mev_per_wallet], axis = 1).any(axis=1).sum()/len(count))