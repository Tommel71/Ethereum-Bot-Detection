import pandas as pd
from tools import load_configs
import os
from tqdm import tqdm
def create_relative_transaction_index(prefix):
    configs = load_configs(prefix)
    prefix_db = configs["General"]["PREFIX_DB"]
    base = f"{prefix_db}/erigon_extract"
    type = "transactions"
    dir = base + "/uncompressed/" + type
    paths = [dir + "/" + path for path in os.listdir(dir)]
    for path in tqdm(paths):
        df = pd.read_csv(path)
        col = "transaction_index"
        newcol = "transaction_index_relative"
        group_key = "block_id"
        # percentage rank
        df[newcol] = df.groupby(group_key)[col].rank(pct=True)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    prefix = ".."
    create_relative_transaction_index(prefix)