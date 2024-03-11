import pandas as pd
from tools import load_configs
import os
from tqdm import tqdm

def run(prefix):
    configs = load_configs(prefix)

    from_folder = f'{configs["General"]["PREFIX_DB"]}/erigon_extract/compressed/logs_old'
    to_folder = f'{configs["General"]["PREFIX_DB"]}/erigon_extract/compressed/logs'

    # create tofolder if it doesnt exist
    if not os.path.exists(to_folder):
        os.makedirs(to_folder)

    df_reference_path = r"E:\Masterthesis\large\erigon_extract\compressed\logs/" + "logs_15513000-15513999.csv.gz"
    df_reference = pd.read_csv(df_reference_path, compression="gzip")
    cols = df_reference.columns

    def transform(new_df):
        # create a new col "transaction_hash", which is the same as "tx_hash"
        new_df["transaction_hash"] = new_df["tx_hash"]
        # remove topic0
        new_df.drop("topic0", axis=1, inplace=True)
        # the field "topics" looks like this ["topic0", "topic1", "topic2", "topic3"] but should be changed to topic0|topic1|topic2|topic3

        new_df["topics"] = new_df["topics"].apply(lambda x: "|".join(x[1:-1].replace('"', "").split(",")))
        # reindex columns to match reference
        new_df = new_df.reindex(columns=cols)
        old_style = new_df
        return old_style


    files = os.listdir(from_folder)
    for file in tqdm(files):
        df = pd.read_csv(f"{from_folder}/{file}", compression="gzip", sep="|")
        df = transform(df)
        # dont use ticks for string
        df.to_csv(f"{to_folder}/{file}", compression="gzip", sep=",", index=False)


if __name__ == "__main__":
    names = ["largesample1", "largesample2", "largesample3", "largesample4"]

    prefix = ".."
    for name in names:
        from tools import set_configs
        set_configs(name, prefix)
        run(prefix)