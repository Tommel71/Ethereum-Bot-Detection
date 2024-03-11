import pandas as pd
from collections import Counter
from scipy.stats import chisquare
import numpy as np
from typing import List
from functools import lru_cache
from math import e
from datetime import datetime
import pyarrow.compute as pc


def calculate_trade_value_clustering(value_list_series: pd.Series) -> pd.DataFrame:
    """
    Clustering occurs because authentic traders tend to use round numbers as cognitive reference points

    Because different tokens have different prices depending on factors like total supply in circulation, demand
    and the number of decimals, which is a technical parameter in the smart contract, it is difficult to compare
    them on their value. We therefore use the first 5 significant digits of the transaction value to compare.
    Note that after this transformation t t(10*x) = t(x) and therefore some clusters will collapse because they
    are multiples of 10 of each other. However, we are mostly interested in unusual numbers in the values
    And this lets us work in a token agnostic way.
    """

    def trade_size_clustering_statistic(values_containing_zeros: np.array) -> float:
        """

        Inspired by Cong 2021

        To quantify the effect of trade-size clustering, we conduct the Student’s t-test for each crypto exchange
         by comparing trade frequencies at round trade sizes with the highest frequency of nearby unrounded trades.
          For each trading pair, we set up two sets of observation windows: windows centered on multiples of 100 units
          (100X) with a radius of 50 units (100X-50, 100X+50), and windows centered on multiples of 500 units (500Y) with
          a radius of 100 units (500Y-100, 500Y+100). Trade frequency is calculated as the number of trades with size i
          over total trade numbers in the observation window


        But we can simplify this by dividing number of overall unrounded trades by number of overall rounded trades

        """

        # we consider the first
        # cut values back to 5 significant digits
        values = values_containing_zeros[values_containing_zeros != 0]

        if len(values) == 0:
            return np.nan
        try:

            values = np.array([int(str(x)[:10]) for x in values])
            mask = values % 100 == 0
        except:
            mask = np.apply_along_axis(lambda x: int(str(x[0])[:10]) % 100 == 0, 1, values.reshape(-1,
                                                                                                  1))  # stupid hack because Decimal can cause problems, check if its too slow later
        rounded_trades = values[mask]
        unrounded_trades = values[~mask]

        rounded_trade_n = len(rounded_trades)
        unrounded_trade_n = len(unrounded_trades)

        return rounded_trade_n / max(1.0, unrounded_trade_n)

    # apply the statistic to the series
    output_series = value_list_series.apply(trade_size_clustering_statistic)

    return pd.DataFrame({"tvc": output_series})


def calculate_benfords_law(value_list_series: pd.Series) -> pd.DataFrame:
    """
    value_list_series is a series that contains lists of integer values.
    Every list contains the set of values to be checked for Benfords law

    We investigate whether the first-significant-digit distribution of transactions on each exchange
    conforms to the pattern implied
     by Benford’s law. Inconsistency with Benford’s law suggests potential manipulations.
     Use Pearson’s Chi-squared test to check the consistency of the first-significant-digit distribution

    """

    @lru_cache(maxsize=10000)
    def get_first_significant_digit(value: int):
        """
        We can assume it's an int, because the values are in wei
        """
        if value == 0:
            return 0
        else:
            return int(str(value)[0])

    @lru_cache(maxsize=1)
    def benfords_law_distribution():
        # Benford's law distribution for first significant digits
        return np.array([np.log10(1 + 1 / d) for d in range(1, 10)])

    def calculate_chi_squared(observed, expected):
        return chisquare(observed, f_exp=expected)

    results = []
    for values in value_list_series:
        # remove 0 values
        values_nozero = values[values != 0]

        observed = np.zeros(9, dtype=int)
        for value in values_nozero:
            first_digit = get_first_significant_digit(value)
            observed[first_digit - 1] += 1

        expected = benfords_law_distribution() * len(values_nozero)
        chi2, p_value = calculate_chi_squared(observed, expected)

        # results.append({'Observed': observed, 'Expected': expected, 'Chi-Square': chi2, 'P-Value': p_value})
        results.append({'benfords': p_value})

    result_df = pd.DataFrame(results)
    return result_df

def aggregate_time_based(series_of_lists: pd.Series) -> pd.DataFrame:
    """
    Extract time-based features.

    series_of_lists contains a list of timestamps for each address. There can be a None instead of a list

    so the series can look like this:

    0 [timestamp1, timestamp2, timestamp3]
    1 [timestamp1, timestamp2, timestamp3, timestamp4]
    2 None
    3 [timestamp1, timestamp2]

    """

    # get highest and lowest time
    low, high = series_of_lists.apply(lambda x: min(x)).min(), series_of_lists.apply(lambda x: max(x)).max()

    # Calculate sleepiness feature
    def calculate_sleepiness(lst):
        """
        you get a list of times. then you split that list into chunks of two days.
        Then you get the timedifferences within the chunks and get the maximum within a chunk.
        now average over chunks
        :param lst:
        :return:
        """
        # Create chunks of two days
        chunks = []

        if lst is np.nan:
            return np.nan

        if len(lst) <= 2:
            return np.nan


        two_days_in_seconds = 2 * 24 * 60 * 60
        chunk_start = lst[0]
        chunk_i = 0
        chunks.append([chunk_start])
        for timestamp in lst[1:]:
            if (timestamp - chunk_start) >= two_days_in_seconds:  # Two days in seconds
                chunk_i += 1
                chunk_start = timestamp
                chunks.append([chunk_start])
            else:
                chunks[chunk_i].append(timestamp)


        # Calculate the time differences and find maximum within each chunk
        max_diffs = []
        for chunk in chunks:
            chunk_diffs = [chunk[i + 1] - chunk[i] for i in range(len(chunk) - 1)]
            if chunk_diffs == []:
                maximum = two_days_in_seconds
            else:
                maximum = max(chunk_diffs)
                maximum = max(maximum, min(high, chunk[0] +  two_days_in_seconds) - chunk[-1]) #  consider right border
            max_diffs.append(maximum)

        # Calculate the average of maximum time differences
        average_max_diff = sum(max_diffs) / len(max_diffs)
        return average_max_diff

    sleepiness = series_of_lists.apply(calculate_sleepiness)

    # Calculate transaction frequency
    def calculate_transaction_frequency(lst):

        if lst is np.nan:
            return np.nan

        if len(lst) < 2:
            return np.nan

        timeframe_length = lst[-1] - lst[0]
        if timeframe_length == 0:
            return np.nan
        freq = len(lst) / timeframe_length
        return freq

    transaction_frequency = series_of_lists.apply(calculate_transaction_frequency)

    def calc_timediff(lst):
        if lst is np.nan:
            return [np.nan] * 6

        if len(lst) < 2:
            return [np.nan] * 6

        timediff = np.diff(lst)
        return timediff

    distribution_details = get_generic_statistics(series_of_lists.apply(calc_timediff))

    """
    We take the list of transaction times and create a bucket for each hour of the day and count the transaction in each bucket. Then we calculate the entropy of this distribution.
    """
    dists = series_of_lists.apply(lambda x: [datetime.fromtimestamp(t).hour for t in x])
    hourly_entropy = dists.apply(entropy)

    # Create the resulting DataFrame
    result_df_1 = pd.DataFrame({
        'sleepiness': sleepiness,
        'transaction_frequency': transaction_frequency,
        'hourly_entropy': hourly_entropy
    })
    result_df = pd.concat([result_df_1, distribution_details],
                          axis=1)
    #result_df.columns = [f"time__{col}" for col in result_df.columns]

    return result_df

def aggregate_generic_features(data: pd.DataFrame, generic_cols_dist, generic_cols_stats) -> pd.DataFrame:
    """
    data is a dataframe which contains grouped data for each address. e.g. if there is a column "gas"
    then in the first row there is a np.array of all gas values for the first address.

    Based on whether the column is in generic_cols_dist or generic_cols_stats, the function calculates
    the distribution or the statistics of the column.
    """

    dfs_dist = []

    for col in generic_cols_dist:
        df_temp = get_generic_categorical(data[col])
        df_temp.columns = [f"{col}_{c}" for c in df_temp.columns]
        dfs_dist.append(df_temp)

    df_dist = pd.concat(dfs_dist, axis=1)

    dfs_stats = []
    for col in generic_cols_stats:
        df_temp = get_generic_statistics(data[col])
        df_temp.columns = [f"{col}_{c}" for c in df_temp.columns]
        dfs_stats.append(df_temp)

    df_stats = pd.concat(dfs_stats, axis=1)
    df_out = pd.concat([df_dist, df_stats], axis=1)

    return df_out

def tx_time_based_in_features(time_transactions_in):
    df_features_in = aggregate_time_based(time_transactions_in["series"])
    df_features_in.columns = [f"time__intime_{col}" for col in df_features_in.columns]
    df_features_in["address"] = time_transactions_in["address"]
    return df_features_in


def tx_time_based_out_features(time_transactions_out):
    df_features_out = aggregate_time_based(time_transactions_out["series"])
    df_features_out.columns = [f"time__outtime_{col}" for col in df_features_out.columns]
    df_features_out["address"] = time_transactions_out["address"]
    return df_features_out


def tx_value_based_features(value_transactions_out):
    # trade value clustering
    trade_value_clustering = calculate_trade_value_clustering(value_transactions_out["series"])
    trade_value_clustering.columns = [f"value__{col}" for col in trade_value_clustering.columns]
    trade_value_clustering["address"] = value_transactions_out["address"]

    # benfords law
    benfords_law = calculate_benfords_law(value_transactions_out["series"])
    benfords_law.columns = [f"value__{col}" for col in benfords_law.columns]
    # benfords_law["address"] = value_transactions_out["address"]

    # concatenate and return
    df_features = pd.concat([trade_value_clustering, benfords_law], axis=1)
    return df_features

def tx_block_based_features(data_df_custom, n_blocks):
    n_tx = data_df_custom.apply(lambda x: len(x["series"]), axis=1).values
    custom_cols = {
        "n_tx_per_block": n_tx / n_blocks,
    }
    # count number of transactions per block
    tx_counts = data_df_custom.apply(lambda x: np.array(list(Counter(x["series"]).values())), axis=1)
    stats_tx_counts = get_generic_statistics(tx_counts)
    stats_tx_counts.columns = [f"tx_per_active_block_{col}" for col in stats_tx_counts.columns]
    custom_features = pd.DataFrame(custom_cols)
    custom_features = pd.concat([custom_features, stats_tx_counts], axis=1)
    custom_features.columns = [f"custom__{col}" for col in custom_features.columns]
    custom_features["address"] = data_df_custom["address"]
    return custom_features

def tx_generic_features(data_df_generic, generic_cols):

    generic_cols_dist, generic_cols_stats = generic_cols["dist"], generic_cols["stats"]
    generic_features = aggregate_generic_features(data_df_generic, generic_cols_dist, generic_cols_stats)
    generic_features.columns = [f"generic__{col}" for col in generic_features.columns]
    generic_features["address"] = data_df_generic["address"]
    return generic_features

def get_generic_categorical(series: pd.Series) -> pd.DataFrame:
    return generic_categorical_to_df(series.apply(generic_categorical))

def get_generic_statistics(series: pd.Series) -> pd.DataFrame:
    return generic_statistics_to_df(series.apply(generic_statistics))

# Calculate distribution details
def generic_statistics(lst: np.array) -> List[float]:
    if lst is np.nan:
        return [np.nan] * 6

    try:
        quantile = pc.quantile(lst, 0.95)[0].as_py()
    except:
        quantile = np.nan

    if quantile is None:
        quantile = np.nan

    output = [np.mean(lst), np.median(lst), np.std(lst), np.min(lst), np.max(lst)] + [quantile]

    return output


def entropy(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    return entropy_from_counts(counts, base=base)

def entropy_from_counts(counts, base=None):
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

def entropy_from_counts_remove_zeros(counts: np.array, base=None):
    counts = counts[counts != 0]
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

def generic_categorical(lst: np.array) -> dict:
    """return percentage of all the values in the list"""
    if lst is np.nan:
        return np.nan

    if len(lst) == 0:
        return np.nan

    dist = Counter(lst)
    dist = {k: v / len(lst) for k, v in dist.items()}
    mode = max(dist, key=dist.get)
    entropy_value = entropy(lst)
    dist = {"mode": mode, "entropy": entropy_value, **dist}
    return dist


def address_features(df):
    addresses = df["address"]
    addresses_str = addresses.astype(str).str[2:]
    # remove first 2 characters (0x)
    # Calculate the number of starting zeros
    # Calculate the number of starting zeros
    features = pd.DataFrame(columns=['starting_zeros'])
    starting_zeros = addresses_str.str.len() - addresses_str.str.lstrip('0').str.len()
    features['starting_zeros'] = starting_zeros
    # fill na with 0 for that col
    features['starting_zeros'] = features['starting_zeros'].fillna(0)
    # cast to int
    features['starting_zeros'] = features['starting_zeros'].astype(int)

    # Calculate digit distribution
    digit_counts = addresses_str.apply(lambda x: Counter(x.replace(r'\D', '')))
    digit_distribution = pd.DataFrame(list(digit_counts))
    digit_distribution.fillna(0, inplace=True)
    digit_distribution = digit_distribution.astype(int)
    # sort the columns
    digit_distribution = digit_distribution.reindex(sorted(digit_distribution.columns), axis=1)
    out_df = pd.concat([features, digit_distribution], axis=1)

    #out_df.columns = [f"address__{col}" if col != "address" else col for col in out_df.columns]
    out_df["address"] = addresses
    return out_df

def generic_categorical_to_df(series: pd.Series) -> pd.DataFrame:
    """
    series contains a list of dictionaries. each dictionary is a distribution.
    :param series:
    :return:
    """
    df = pd.DataFrame(series.tolist())
    df.fillna(0, inplace=True)
    # sort columns
    df.columns = df.columns.astype(str)
    df = df.reindex(sorted(df.columns), axis=1)

    return df


def generic_statistics_to_df(dist_data) -> pd.DataFrame:
    out_df = pd.DataFrame({
        'mean': dist_data.apply(lambda x: x[0]),
        'median': dist_data.apply(lambda x: x[1]),
        'std': dist_data.apply(lambda x: x[2]),
        'min': dist_data.apply(lambda x: x[3]),
        'max': dist_data.apply(lambda x: x[4]),
        'quantile_95': dist_data.apply(lambda x: x[5])
    })

    return out_df


def value_features(data: pd.Series) -> pd.DataFrame:
    bl = calculate_benfords_law(data)
    tvc = calculate_trade_value_clustering(data)
    stats = get_generic_statistics(data)
    df_out = pd.concat([bl.reset_index(drop=True), tvc.reset_index(drop=True), stats.reset_index(drop=True)], axis=1)
    df_out.index = data.index
    return df_out

def swap_based_features(df_i, amount_cols, i, n_blocks):
    dfs_i = []
    amount_cols = [col.lower() for col in amount_cols]

    for col in amount_cols:
        data = df_i[col]
        df_col = value_features(data)
        df_col.columns = [f"{i}__{col}__{colname}" for colname in df_col.columns]
        dfs_i.append(df_col)


    df_path_length_stats = get_generic_statistics(df_i["pathlength"])
    df_path_length_stats.columns = [f"{i}__pathlength__{colname}" for colname in df_path_length_stats.columns]
    dfs_i.append(df_path_length_stats)

    df = pd.concat([df_i.address] + dfs_i, axis=1)

    n_tx = df_i[[amount_cols[0]]].apply(lambda x: len(x[amount_cols[0]]), axis=1).values
    df[f"{i}__swaps_per_block"] = n_tx / n_blocks


    return df


def erc20_transfer_features(df, n_blocks):
    """
    Apply value features to the "value" column
    :param df:
    :return:
    """
    amount_col = "value"

    data = df[amount_col]
    df_col = value_features(data)
    df_col.columns = [f"{amount_col}__{colname}" for colname in df_col.columns]
    return_df = pd.concat([df.address, df_col], axis=1)


    n_tx = df.apply(lambda x: len(x[amount_col]), axis=1).values
    return_df["transfers_per_block"] = n_tx / n_blocks
    return return_df

def event_swap_features(df, n_blocks):

    amount_col = "amountin"
    data = df[amount_col]
    df_col = value_features(data)
    df_col.columns = [f"{amount_col}__{colname}" for colname in df_col.columns]
    return_df = pd.concat([df.address, df_col], axis=1)

    n_tx = df.apply(lambda x: len(x[amount_col]), axis=1).values
    return_df["swaps_per_block"] = n_tx / n_blocks
    return return_df