import pandas as pd
from src.Aggregate import Aggregate
from tools import load_configs
from tools import save_data_for_figure
from tools import save_text

def get_relevant_addresses(df):
    block_col = "blockNumber"
    # make sure blocknumber is from 15500000 to 15599999
    df = df[(df[block_col] >= 15500000) & (df[block_col] < 15600000)]
    from_addresses = df["from"].unique()
    return list(from_addresses)


def run(prefix):
    eigenphi_path_arb = f"{prefix}/data/eigenphi/arbitrage_all.csv"
    eigenphi_path_liq = f"{prefix}/data/eigenphi/liquidation_all.csv"
    eigenphi_path_sand = f"{prefix}/data/eigenphi/sandwich_all.csv"

    configs = load_configs(prefix)

    agg = Aggregate(configs, prefix=prefix)
    df_arb, df_liq, df_sand = agg.load_MEVinspect("arbitrages"), agg.load_MEVinspect(
        "liquidations"), agg.load_MEVinspect("sandwiches")
    # load and ignore whitespace in cols and data
    df_arb_eig = pd.read_csv(eigenphi_path_arb, skipinitialspace=True)

    df_liq_eig = pd.read_csv(eigenphi_path_liq, skipinitialspace=True)
    df_sand_eig = pd.read_csv(eigenphi_path_sand, skipinitialspace=True)

    df_arb_addresses_eig = get_relevant_addresses(df_arb_eig)
    df_liq_addresses_eig = get_relevant_addresses(df_liq_eig)
    df_sand_eig.rename(columns={"attackerEOA": "from"}, inplace=True)
    df_sand_addresses_eig = get_relevant_addresses(df_sand_eig)


    # Calculate the number of unique addresses for each type in MEV-Inspect
    num_arb_addresses_mev = len(df_arb.index)
    num_liq_addresses_mev = len(df_liq.index)
    num_sand_addresses_mev = len(df_sand.index)

    # Calculate the number of unique addresses for each type in Eigenphi
    num_arb_addresses_eig = len(df_arb_addresses_eig)
    num_liq_addresses_eig = len(df_liq_addresses_eig)
    num_sand_addresses_eig = len(df_sand_addresses_eig)

    # Calculate the number of common addresses between MEV-Inspect and Eigenphi for each type
    common_arb_addresses = len(set(df_arb.index) & set(df_arb_addresses_eig))
    common_liq_addresses = len(set(df_liq.index) & set(df_liq_addresses_eig))
    common_sand_addresses = len(set(df_sand.index) & set(df_sand_addresses_eig))

    # Data for the bar plot
    labels = ['Arbitrages', 'Liquidations', 'Sandwiches']
    mev_counts = [num_arb_addresses_mev, num_liq_addresses_mev, num_sand_addresses_mev]
    eig_counts = [num_arb_addresses_eig, num_liq_addresses_eig, num_sand_addresses_eig]
    common_counts = [common_arb_addresses, common_liq_addresses, common_sand_addresses]

    data = labels, mev_counts, eig_counts, common_counts
    figurename = "mevinspect_vs_eigenphi_addresses"
    chapter = "background"
    save_data_for_figure(data, figurename, chapter, prefix)

    # now do the same for the number of transactions.
    df_arb_raw = pd.read_csv(f"{prefix}/data/mev_inspect_predictions/arbitrages.csv")
    df_liq_raw = pd.read_csv(f"{prefix}/data/mev_inspect_predictions/liquidations.csv")
    df_sand_raw = pd.read_csv(f"{prefix}/data/mev_inspect_predictions/sandwiches.csv")

    arb_tx_mi = df_arb_raw["tx_hash"]
    liq_tx_mi = df_liq_raw["tx_hash"]
    sand_tx_mi = df_sand_raw["tx_hash"]

    arb_tx_eig = df_arb_eig[(df_arb_eig["blockNumber"] >= 15500000) & (df_arb_eig["blockNumber"] < 15600000)]["txHash"]
    liq_tx_eig = df_liq_eig[(df_liq_eig["blockNumber"] >= 15500000) & (df_liq_eig["blockNumber"] < 15600000)][
        "transactionHash"]
    sand_tx_eig = df_sand_eig[(df_sand_eig["blockNumber"] >= 15500000) & (df_sand_eig["blockNumber"] < 15600000)][
        "attackerTxs"]
    # split for sand on space
    sand_tx_eig = sand_tx_eig.str.split(" ")
    sand_tx_eig = [item for sublist in sand_tx_eig for item in sublist]

    # Calculate the number of unique transactions for each type in MEV-Inspect
    num_arb_tx_mev = len(arb_tx_mi)
    num_liq_tx_mev = len(liq_tx_mi)
    num_sand_tx_mev = len(sand_tx_mi)

    # Calculate the number of unique transactions for each type in Eigenphi
    num_arb_tx_eig = len(arb_tx_eig)
    num_liq_tx_eig = len(liq_tx_eig)
    num_sand_tx_eig = len(sand_tx_eig)

    # Calculate the number of common transactions between MEV-Inspect and Eigenphi for each type
    common_arb_tx = len(set(arb_tx_mi) & set(arb_tx_eig))
    common_liq_tx = len(set(liq_tx_mi) & set(liq_tx_eig))
    common_sand_tx = len(set(sand_tx_mi) & set(sand_tx_eig))

    sand_tx_only_mi = set(sand_tx_mi) - set(sand_tx_eig)
    df_sand_tx = agg.load_MEVinspect_transactions("sandwiches")
    df_sand_tx_filtered = df_sand_tx[df_sand_tx["tx_hash"].isin(sand_tx_only_mi)]
    unique_only_mi_addressbased = df_sand_tx_filtered["from_address"].unique()
    diff_addresses_mi_ep = set(df_sand.index) - set(df_sand_addresses_eig)
    addressdiff_caused_by_those_txes = set(unique_only_mi_addressbased).intersection(diff_addresses_mi_ep)

    # NOTE: THESE ARE NOT NECESSARY AS IT SHOULD BE POSSIBLE TO SEE THEM IN THE BAR PLOTS
    text, filename, chapter, prefix = len(df_sand_tx_filtered), "sandwich_txes_only_mi", "background", prefix
    save_text(text, filename, chapter, prefix)
    text, filename, chapter, prefix = len(addressdiff_caused_by_those_txes), "addressdiff_caused_by_those_txes", "background", prefix
    save_text(text, filename, chapter, prefix)

    # Data for the bar plot
    labels = ['Arbitrages', 'Liquidations', 'Sandwiches']
    visibility_factor = 2
    mev_counts = [num_arb_tx_mev, visibility_factor * num_liq_tx_mev, num_sand_tx_mev]
    eig_counts = [num_arb_tx_eig, visibility_factor * num_liq_tx_eig, num_sand_tx_eig]
    common_counts = [common_arb_tx, visibility_factor * common_liq_tx, common_sand_tx]

    # Create the bar plot
    data = labels, mev_counts, eig_counts, common_counts
    figurename = "mevinspect_vs_eigenphi_transactions"
    chapter = "background"
    save_data_for_figure(data, figurename, chapter, prefix)



if __name__ == "__main__":
    prefix =".."
    run(prefix)
