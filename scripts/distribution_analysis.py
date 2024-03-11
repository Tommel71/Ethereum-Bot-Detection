from matplotlib import pyplot as plt
import numpy as np
import requests
import pandas as pd
from tqdm import tqdm
from tools import load_json
import os
import web3
from web3 import Web3
from tools import get_web3

prefix = ".."
addresses = pd.read_csv(prefix + "/data/wallets_to_annotate.csv", header=None)[0].values

data = []

n_tx_list = []
for address in addresses:
    address = address.lower()
    df = pd.read_pickle(f"{prefix}/data/etherscan/{address}.pkl")
    n_tx = df.shape[0]
    n_tx_list.append(n_tx)


etherscan_api_token = list(load_json(f"{prefix}/credentials/etherscan.json").values())[0]

# query all from addresses in block 15000000

# download block from etherscan
startblock = 15000000
blocks = list(range(startblock, startblock + 1000))
df_list = []
block_time_list = []
scraped = False

if os.path.exists(f"{prefix}/data/etherscan_distribution/blocks.csv") and os.path.exists(f"{prefix}/data/etherscan_distribution/n_tx.csv"):
    scraped = True

# scraped on 25.1.2023
if not scraped:
    for block in tqdm(blocks):

        get = f"http://api.etherscan.io/api?module=proxy&action=eth_getBlockByNumber&tag={hex(block)}&boolean=true&apikey={etherscan_api_token}"

        r = requests.get(get)
        data = r.json()["result"]
        df = pd.DataFrame(data["transactions"])
        df_list.append(df)

    df = pd.concat(df_list)
    addresses = df["from"].unique()
    n_tx_list = []
    for address in tqdm(addresses):
        query = f"http://api.etherscan.io/api?module=proxy&action=eth_getTransactionCount&address={address}&tag=latest&apikey={etherscan_api_token}"
        r = requests.get(query)
        n_tx = int(r.json()["result"], 16)
        n_tx_list.append(n_tx)
        print(n_tx)

    df_n_tx = pd.DataFrame({"address": addresses, "n_tx": n_tx_list})
    df_n_tx.to_csv(f"{prefix}/data/etherscan_distribution/n_tx.csv", index=False)
    df.to_csv(f"{prefix}/data/etherscan_distribution/blocks.csv", index=False)

else:
    df = pd.read_csv(f"{prefix}/data/etherscan_distribution/blocks.csv")
    df_n_tx = pd.read_csv(f"{prefix}/data/etherscan_distribution/n_tx.csv", index_col="address")



mask = df_n_tx["n_tx"] < 10000
df_n_tx[mask].hist(bins=100)
plt.xlabel("Number of transactions")
plt.ylabel("Number of addresses")
plt.show()




df_n_tx_log = df_n_tx.copy()
df_n_tx_log["n_tx"] = np.log10(df_n_tx_log["n_tx"])
df_n_tx_log.hist(bins=300)
plt.xlabel("log(Number of transactions)")
plt.ylabel("Number of addresses")
plt.show()


# use web3 to get the timestamps of all blocks in 10 days

# get the blocktime of the blocks

# load
w3 = get_web3(prefix)
# get the timestamps of 10000 blocks


# UTC time
first_block_after_24 = 14999391
last_block_before_24_2_days_later = 15010210
blocks_for_time = list(range(first_block_after_24, last_block_before_24_2_days_later+1))
block_time_list = []
n_tx_list = []

if os.path.exists(f"{prefix}/data/etherscan_distribution/block_time.csv"):
    for block in tqdm(blocks_for_time):
        block_obj = w3.eth.getBlock(block)
        block_time = block_obj["timestamp"]
        n_tx = len(block_obj["transactions"])

        block_time_list.append(block_time)
        n_tx_list.append(n_tx)

    df_block_time = pd.DataFrame({"block": blocks_for_time, "block_time": block_time_list, "n_tx": n_tx_list})
    df_block_time.to_csv(f"{prefix}/data/etherscan_distribution/block_time.csv", index=False)

else:
    df_block_time = pd.read_csv(f"{prefix}/data/etherscan_distribution/block_time.csv", index_col="block")

# one hour of the day is one bin, sum up the number of transactions in each bin
df_block_time["hour"] = df_block_time["time"] % (24*60*60)
df_block_time["hour"] = df_block_time["hour"] // (60*60)
df_block_time["n_tx"] = n_tx_list

df_block_time.groupby("hour")["n_tx"].sum().plot.bar()
plt.xlabel("Hour of the day")
plt.ylabel("Number of transactions")
plt.show()
