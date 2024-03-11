import pandas as pd
from src.Labeling.annotation_macro import find_out_facts
from tools import save_data_for_figure, save_table

def inspect(address, prefix):

    address = address.lower()
    df = pd.read_pickle(f"{prefix}/data/etherscan/{address}.pkl")
    facts, df = find_out_facts(address, df) # main feature generation here
    df_print = pd.DataFrame([facts]).transpose().reset_index()
    df_print.columns = ["", ""]
    save_table(df_print, f"{address}_facts", "data", prefix, header=False)
    return df

def inspect_specific(prefix):
    address = "0x478b0660F7F2301F01864C6D2c9111d63Dc65FFC"#"0xD5Ed772FaD590f3aB6bE1Af795E3D3086c113f8d"
    df = inspect(address, prefix)
    save_data_for_figure(df, f"daydist_{address}", "data", prefix)
    save_data_for_figure(df, f"functiondist_{address}", "data", prefix)

if __name__ == "__main__":
    prefix = "../.."
    #address = "0x478b0660F7F2301F01864C6D2c9111d63Dc65FFC" #"0xD5Ed772FaD590f3aB6bE1Af795E3D3086c113f8d"
    inspect_specific(prefix)