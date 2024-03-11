from tools import load_json
import json
from tools import save_json
import math
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from tools import load_block
from mev_inspect.classifiers.trace import TraceClassifier
from mev_inspect.sandwiches import get_sandwiches
from mev_inspect.liquidations import get_liquidations
from mev_inspect.arbitrages import get_arbitrages
from mev_inspect.swaps import get_swaps
import logging
import time
from joblib import Parallel, delayed

logger = logging.getLogger('preprocessing.create_mev_inspect_predictions')

def process_block(block_file):
    data_sandwich_temp = []
    data_liquidations_temp = []
    data_arbitrages_temp = []
    block = load_block(block_file)

    trace_classifier = TraceClassifier()
    classified_traces = trace_classifier.classify(block.traces)

    swaps = get_swaps(classified_traces)
    sandwiches = get_sandwiches(list(swaps))

    if sandwiches:
        for sandwich in sandwiches:
            front_tx = sandwich.frontrun_swap.transaction_hash
            front_tx_block = sandwich.frontrun_swap.block_number

            back_tx = sandwich.backrun_swap.transaction_hash
            back_tx_block = sandwich.backrun_swap.block_number

            data_sandwich_temp += [[front_tx_block, front_tx]]
            data_sandwich_temp += [[back_tx_block, back_tx]]

    liquidations = get_liquidations(classified_traces)
    if liquidations:
        for liquidation in liquidations:
            tx = liquidation.transaction_hash
            block_number = liquidation.block_number
            data_liquidations_temp += [[block_number, tx]]

    arbitrages = get_arbitrages(swaps)
    if arbitrages:
        for arbitrage in arbitrages:
            tx = arbitrage.transaction_hash
            block_number = arbitrage.block_number
            data_arbitrages_temp += [[block_number, tx]]

    return data_sandwich_temp, data_liquidations_temp, data_arbitrages_temp



def csv_to_erigon_folder(trace_folder, logs_folder, erigon_folder, reference_traces_path):
    """
    This function takes the traces and logs from the erigon database and puts them into the format that mev-inspect expects
    For now logs arent actually used. Maybe never necessary.

    :param trace_folder:
    :param logs_folder:
    :param erigon_folder:
    :param reference_traces_path:
    :return:
    """

    traces = os.listdir(trace_folder)
    logs = os.listdir(logs_folder)

    # log files begin with logs trace file begin with trace
    # group the logs and traces based on their number

    # get the number of the trace file
    trace_numbers = [int(trace.split("_")[1].split("-")[0]) for trace in traces]
    # get the number of the log file
    log_numbers = [int(log.split("_")[1].split("-")[0]) for log in logs]

    overlap = set(trace_numbers).intersection(set(log_numbers))
    eligible_trace_files = [trace for trace in traces if int(trace.split("_")[1].split("-")[0]) in overlap]
    eligible_log_files = [log for log in logs if int(log.split("_")[1].split("-")[0]) in overlap]

    # sort the files based on their number
    eligible_trace_files.sort(key=lambda x: int(x.split("_")[1].split("-")[0]))
    eligible_log_files.sort(key=lambda x: int(x.split("_")[1].split("-")[0]))

    reference_traces = pd.DataFrame(load_json(reference_traces_path)["traces"])

    # unwrap the action attribute
    added_columns = pd.json_normalize(reference_traces["action"])
    reference_traces = reference_traces.join(added_columns)
    reference_traces_sorted = reference_traces.reindex(sorted(reference_traces.columns), axis=1)

    reference_traces_sorted.drop(columns=["action"], inplace=True)
    reference_traces_sorted.drop(columns=["block_hash"], inplace=True)

    reference_traces_sorted_unpacked = reference_traces_sorted
    reference_traces_sorted_unpacked["gasUsed"] = [x["gasUsed"] if x is not None else None for x in
                                         reference_traces_sorted_unpacked["result"]]
    reference_traces_sorted_unpacked["output"] = [x["output"] if x is not None else None for x in
                                        reference_traces_sorted_unpacked["result"]]
    target_columns = pd.DataFrame(reference_traces_sorted_unpacked).drop(columns=["result"]).columns

    for trace_file, log_file in tqdm(zip(eligible_trace_files, eligible_log_files)):
        csv_to_erigon(os.path.join(trace_folder, trace_file),
                        erigon_folder,
                        target_columns)


def csv_to_erigon(trace_csv: str, erigon_folder: str, target_columns):
    """

    :param trace_csv:
    :param erigon_folder:
    :param target_columns: Which columns we should have in the resulting json file. Uses reference json from mev-inspect
    :return:
    """


    df_traces = pd.read_csv(trace_csv)
    trace_name_mapping = {"block_id": "block_number", "call_type": "callType", "from_address": "from",
                          "to_address": "to", "gas_used": "gasUsed", "trace_type": "type", "tx_hash": "transaction_hash"
        , "transaction_index": "transaction_position", "reward_type": "rewardType"}


    df_sorted = df_traces.reindex(sorted(df_traces.columns), axis=1)




    df_renamed = df_sorted.rename(columns=trace_name_mapping)

    # remove overlapping columnnames of df_sorted and reference_traces_sorted
    df_sorted_diff = df_renamed.loc[:, ~df_renamed.columns.isin(target_columns)]
    #reference_traces_sorted_diff = reference_traces_sorted_unpacked.loc[:, ~reference_traces_sorted_unpacked.columns.isin(df_renamed.columns)]

    ### receiptS

    # dd is a dict, unpack dd["action"]
    #dd_receipts = pd.read_csv(log_csv) # logs != receipts, dont have receipts
    #df_sorted_receipts = dd_receipts.reindex(sorted(dd_receipts.columns), axis=1)
    #df_mi_dict_receipts = load_json(f"{self.prefix}/data/mev_inspect_test.json")
    #reference_traces_receipts = pd.DataFrame(df_mi_dict_receipts["receipts"])

    # DUMMIES
    receipts = [{"blockNumber": 1,  # dummy
                 "transaction_hash": "0x123",
                 "transaction_index": 1,
                 "gas_used": 1,
                 "effective_gas_price": 1,
                 "cumulative_gas_used": 1,
                 "to": "0x123"
                 }]

    # this part of the traces is only missing if its a suicide
    # for now its just always nan
    missing_part_traces = {  # dummy
        "address": np.nan,
        "author": np.nan,
        "balance": np.nan,
        "refundAddress": np.nan
    }

    columns_to_drop_traces = df_sorted_diff.columns

    df_traces_renamed = df_traces.rename(columns=trace_name_mapping)

    # add missing parts of the trace
    df_traces_full = pd.concat(
        [df_traces_renamed, pd.DataFrame([missing_part_traces for _ in range(len(df_traces_renamed))])], axis=1)

    def to_hex(x):

        if type(x) is str:
            x = int(x)

        if math.isnan(x):
            return x

        return hex(int(x))

    # drop whats too much
    df_traces_dropped = df_traces_full.drop(columns=columns_to_drop_traces)

    # add a dummy blockhash
    df_traces_dropped["block_hash"] = df_traces_dropped["block_number"].apply(hash).apply(to_hex)
    df_traces_dropped_sorted = df_traces_dropped.reindex(sorted(df_traces_dropped.columns), axis=1)

    # cast to same datatype
    df_traces_dropped_sorted["error"][df_traces_dropped_sorted["error"].isna()] = "None"

    def stroke_to_list(x):

        if type(x) is str:
            return [int(t) for t in x.split("|")]

        if math.isnan(x):
            return []

    df_traces_dropped_sorted["gas"] = df_traces_dropped_sorted["gas"].apply(to_hex)
    df_traces_dropped_sorted["gasUsed"] = df_traces_dropped_sorted["gasUsed"].apply(to_hex)
    df_traces_dropped_sorted["value"] = df_traces_dropped_sorted["value"].apply(to_hex)
    df_traces_dropped_sorted["trace_address"] = df_traces_dropped_sorted["trace_address"].apply(stroke_to_list)
    df_traces_dropped_sorted["transaction_position"] = df_traces_dropped_sorted["transaction_position"].astype(
        pd.Int64Dtype())

    # wrap actions and results
    df_traces_dropped_sorted["action"] = df_traces_dropped_sorted.apply(
        lambda x: {"callType": x["callType"], "from": x["from"], "gas": x["gas"], "input": x["input"], "to": x["to"],
                   "value": x["value"]}, axis=1)
    df_traces_dropped_sorted["result"] = df_traces_dropped_sorted.apply(
        lambda x: {"gasUsed": x["gasUsed"], "output": x["output"]}, axis=1)

    def isnan(x):
        if type(x) is str:
            return False

        return math.isnan(x)

    mask = [isnan(x["gasUsed"]) and isnan(x["output"]) for x in df_traces_dropped_sorted["result"]]
    df_traces_dropped_sorted["result"][mask] = None
    # drop the columns that are now in the action and output
    df_traces_dropped_sorted = df_traces_dropped_sorted.drop(
        columns=["callType", "from", "gas", "input", "to", "value", "gasUsed", "output"])

    block_dfs = df_traces_dropped_sorted.groupby("block_number")

    for block_number in block_dfs.groups.keys():
        print(block_number)
        block_df = block_dfs.get_group(block_number)

        traces = json.loads(block_df.to_json(orient="records"))

        block = {"block_number": block_number,
                 "miner": "0x123",  # dummy
                 "base_fee_per_gas": 123,  # dummy
                 "traces": traces,
                 "receipts": receipts
                 }
        save_json(block, f"{erigon_folder}/{block_number}.json")


def precompute_mev_inspect(blocks_folder, predictions_folder):
    """
    ATTENTION: CANT RUN THIS WITH PYTHON CONSOLE IN PYCHARM DUE TO A BUG
    :param blocks_folder:
    :param predictions_folder:
    :return:
    """
    # for each block in blocks_folder
    # run mev-inspect
    # save the output in predictions_folder

    # if predictions_folder doesnt exist, create it
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)

    data_sandwich = []
    data_liquidations = []
    data_arbitrages = []
    block_files = [f"{blocks_folder}/{block_file}" for block_file in os.listdir(blocks_folder)]


    # now do the same thing in parallel

    start = time.time()
    logger.debug("starting parallel")
    results = Parallel(n_jobs=6)(delayed(process_block)(block_file) for block_file in block_files)


    logger.debug(f"finished parallel in {time.time() - start}")


    for result in results:
        data_sandwich += result[0]
        data_liquidations += result[1]
        data_arbitrages += result[2]


    df_sandwich = pd.DataFrame(data_sandwich, columns=["block_number", "tx_hash"])
    df_liquidations = pd.DataFrame(data_liquidations, columns=["block_number", "tx_hash"])
    df_arbitrages = pd.DataFrame(data_arbitrages, columns=["block_number", "tx_hash"])

    df_sandwich.to_csv(f"{predictions_folder}/sandwiches.csv", index=False)
    df_liquidations.to_csv(f"{predictions_folder}/liquidations.csv", index=False)
    df_arbitrages.to_csv(f"{predictions_folder}/arbitrages.csv", index=False)

