# get one normal block 15599998 and one mev block 15599999 at the end of the observed range
from tools import get_web3
import pandas as pd
from tqdm import tqdm

def get_wallets(blocks, outfile, prefix):
    wallets = []
    # get from provider
    w3 = get_web3(prefix)
    # use web3 to get all transactions in the blocks
    for block in tqdm(blocks):
        block_obj = w3.eth.getBlock(block)

        # get all transactions
        for transaction in block_obj["transactions"]:
            tx_obj = w3.eth.getTransaction(transaction)
            wallets.append({"fromAddress": tx_obj["from"], "blockNumber": block, "txHash": transaction})

    df = pd.DataFrame(wallets)[["fromAddress"]].drop_duplicates()
    df.to_csv(f"{prefix}/{outfile}", index=False, header=None)
    print("finished")

if __name__ == "__main__":
    prefix = ".."
    blocks = [15599998, 15599999]
    get_wallets(blocks, "data/wallets_to_annotate.csv", prefix)
