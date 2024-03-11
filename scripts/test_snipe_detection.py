import pandas as pd
from tools import load_block

for i in range(15000000, 15000020):
    prefix = "."
    path = f"{prefix}/data/mev_inspect_blocks/{i}.json"
    block = load_block(path)
    traces = block.traces
    df = pd.DataFrame(t.action for t in traces)[:-1]
    mask = [t[:10] == "0xc9c65396" for t in df["input"]]
    print(df[mask])