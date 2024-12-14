# query etherscan data and check the usage around the day
import pandas as pd
import requests
import time
prefix = "."
addresses = pd.read_csv(prefix + "/data/wallets_to_annotate.csv", header=None)[0].values


for address in addresses:
    print(address)
    get = f"http://api.etherscan.io/api?module=account&action=txlist&address={address}&sort=asc"
    r = requests.get(get)
    data = r.json()["result"]
    df = pd.DataFrame(data)
    df.to_pickle(f"{prefix}/data/etherscan/{address}.pkl")
    time.sleep(30)

