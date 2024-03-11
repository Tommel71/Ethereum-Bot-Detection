import os
from tqdm import tqdm
import pandas as pd


folder = r"E:\create_all\zipped"
to_folder = r"E:\create_all\merged"
files = os.listdir(folder)
files = [os.path.join(folder, file) for file in files if file.endswith(".csv.gz")]
# for each file read it in, get the columns [code_id, block_number, from_address, to_address, output]
# get only the first 5 characters of the output

dfs = []
for file in tqdm(files):
    df = pd.read_csv(file, compression="gzip")
    df["output"] = df["output"].apply(lambda x: str(x)[:5])
    df = df[["block_number", "from_address", "to_address", "output"]]
    dfs.append(df)


df = pd.concat(dfs)
df.reset_index(drop=True, inplace=True)
df.index.name = "code_id"
df.reset_index(inplace=True)
df.to_csv(os.path.join(to_folder, "trace_creations.csv"), index=False)
outfilename ="trace_creations.csv.gz"
df.to_csv(os.path.join(to_folder, outfilename), index=False, compression="gzip")