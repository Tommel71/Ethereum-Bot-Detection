# If there is no data available, first execute download_etherscan.py to use this
import pandas as pd
import numpy as np

def get_address_interaction(address: str, df: pd.DataFrame):
    if (df["to"].apply(lambda x: x.lower()) == address.lower()).any():
        return "to"
    if (df["from"].apply(lambda x: x.lower()) == address.lower()).any():
        return "from"
    return ""

def find_out_facts(address, df, special_wallets_to_monitor=()):
    """
    Find out facts about the address

    :param address:
    :param df:
    :param special_wallets_to_monitor:
    :return: facts, df
    """

    df["out"] = df["from"] == address

    df["timeStamp"] = pd.to_datetime(df["timeStamp"], unit="s")
    df["hour"] = df["timeStamp"].dt.hour
    df["minute"] = df["timeStamp"].dt.minute
    df["second"] = df["timeStamp"].dt.second
    df["time"] = df["hour"] + df["minute"] / 60 + df["second"] / 3600
    # make sure every hour is represented
    df["time"] = df["time"].round(2)

    df["transactionIndex"] = df["transactionIndex"].astype(int)

    facts = {}
    facts["Address"] = address
    facts["Average Transaction Index"] = df["transactionIndex"].mean()
    facts["N Self Transactions"] = np.logical_and(df["to"] == address, df["from"] == address).sum()
    facts["N Transactions"] = df.shape[0]
    facts["Timeframe traded in"] = df["timeStamp"].max() - df["timeStamp"].min()

    if df["timeStamp"].max() - df["timeStamp"].min() == pd.Timedelta(0, unit="s"):
        facts["Average Transactions per Day"] = None
    else:
        facts["Average Transactions per Day"] = df.shape[0] / (
                df["timeStamp"].max().timestamp() - df["timeStamp"].min().timestamp()) * 3600 * 24

    agg_df = df.groupby(["blockNumber", "out"]).count().reset_index()
    agg_df.set_index("blockNumber", inplace=True)
    mask_out = agg_df["out"] == True


    if agg_df["timeStamp"][mask_out].shape[0] == 0:
        facts["Max same-block Out-Transactions"] = 0
    else:
        masked = agg_df["timeStamp"][mask_out]
        facts["Max same-block Out-Transactions"] = str(masked.max()) + " in block " + str(masked.idxmax())
    if agg_df["timeStamp"][~mask_out].shape[0] == 0:
        facts["Max same-block In-Transactions"] = 0
    else:
        masked = agg_df["timeStamp"][~mask_out]
        facts["Max same-block In-Transactions"] = str(masked.max()) + " in block " + str(masked.idxmax())

    for special_wallet in special_wallets_to_monitor:
        name, address = special_wallet
        facts[name] = get_address_interaction(address, df)

    return facts, df


if __name__ == "__main__":

    prefix = "../.."
    addresses = pd.read_csv(prefix + "/data/wallets_to_annotate.csv", header=None)[0].values
    data = []

    special_wallets_to_monitor = [
        ("quaalude", "0x8FFdD83b2C1541e661c2437B1887844654E050c9"),
        ("stake", "0x974CaA59e49682CdA0AD2bbe82983419A2ECC400"),
        ("disperse", "0xD152f549545093347A162Dce210e7293f1452150")
    ]

    for address in addresses:
        address = address.lower()
        df = pd.read_pickle(f"{prefix}/data/etherscan/{address}.pkl")
        if df.empty:
            continue
        facts, _ = find_out_facts(address, df, special_wallets_to_monitor)

        data.append(facts)


    df = pd.DataFrame(data)
    # move select columns first
    first = ["Average Transactions per Day", "Max same-block Out-Transactions",
             "Max same-block In-Transactions", "N Self Transactions","N Transactions" ,
             "Timeframe traded in", "Average Transaction Index"]
    df = df[first + [col for col in df.columns if col not in first]]
    # df.to_clipboard(index=False)
    print(df)