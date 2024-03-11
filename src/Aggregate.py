import pandas as pd
from tools import load_configs
from src.Datamodels.PipelineComponent import PipelineComponent
import numpy as np
from typing import List
from datetime import datetime
from tools import psql_insert_df, log_time
from multiprocessing import Pool
import os
from src.Aggregating.features import (
    tx_time_based_in_features,
    tx_time_based_out_features,
    tx_value_based_features,
    tx_generic_features,
    tx_block_based_features,
    address_features,
    swap_based_features,
    erc20_transfer_features,
    event_swap_features
)

class Aggregate(PipelineComponent):


    def get_df(self, query: str) -> pd.DataFrame:

        self.connect_databases()
        self.cur.execute(query)

        column_names = [desc[0] for desc in self.cur.description]
        df = pd.DataFrame(data=self.cur.fetchall(), columns=column_names)
        self.disconnect_databases()
        return df


    def get_df_generator(self, query: str):
        """
        This function is a monster but it works

        :param query:
        :param columnnames:
        :return:
        """

        cur = self.get_extra_cursor()
        self.logger.debug(f"Executing query {query}")
        cur.execute(query)
        column_names = [desc[0] for desc in cur.description]

        fetch_n = 100_000
        class DfIterator:

            def rows_to_dataframes(me):
                # check for all cols if they are lists
                # if so, convert them to arrays
                fetched = cur.fetchmany(fetch_n)
                if len(fetched) == 0:
                    return

                df = pd.DataFrame(data=fetched, columns=column_names)
                for col in df.columns:
                    if isinstance(df[col].iloc[0], list):
                        df[col] = df[col].apply(lambda x: np.array(sorted(x)))
                return df

            def __next__(me):
                ret = me.rows_to_dataframes()
                if ret is None:
                    cur.close()
                    raise StopIteration()
                return ret

            def __iter__(me):
                return me


        return DfIterator()



    def group_time_based_postgre_out(self):

        # Execute the SQL query to aggregate the data
        query = """
            SELECT t.from_address as address, array_agg(b.timestamp) AS series
            FROM transactions t
            JOIN blocks b ON t.block_id = b.block_id
            GROUP BY t.from_address
        """

        return self.get_df_generator(query)

    def group_time_based_postgre_in(self):

        # Execute the SQL query to aggregate the data
        query = """
            SELECT t.to_address as address, array_agg(b.timestamp) AS series
            FROM transactions t
            JOIN blocks b ON t.block_id = b.block_id
            GROUP BY t.to_address
        """

        return self.get_df_generator(query)



    def group_value_based_postgre(self):
        """
        Group transactions by value
        :return:
        """

        query = """
            SELECT from_address as address, array_agg(value) AS series
            FROM transactions
            GROUP BY from_address
        """

        return self.get_df_generator(query)


    def group_tx_generic_postgre(self):
        """
        Group transactions by value
        :return:
        """

        generic_cols = self.configs["Aggregate"]["tx_generic_cols"]
        generic_cols_dist, generic_cols_stats = generic_cols["dist"], generic_cols["stats"]
        generic_cols_dist_str = ", ".join([f"array_agg({col}) AS {col}" for col in generic_cols_dist])
        generic_cols_stats_str = ", ".join([f"array_agg({col}) AS {col}" for col in generic_cols_stats])
        generic_cols_str = ", ".join([generic_cols_dist_str, generic_cols_stats_str])

        query = f"""
            SELECT from_address as address, {generic_cols_str} FROM transactions
            GROUP BY from_address
        """

        return self.get_df_generator(query)

    @log_time
    def address_based_features(self) -> List[str]:
        """
        Features based on the actual digits of the address. For example the number of starting 0s in the address.
        The distribution of the digits in the address.
        """

        addresses_gen = self.get_addresses()
        table_name = self.process_piecewise(address_features, addresses_gen,
                                                  (), "address",
                                                  "address_based_features")

        return [table_name]


    def delete_table_if_exists(self, tablename, cascade=True):
        self.connect_databases()
        if cascade:
            self.cur.execute(f"DROP TABLE IF EXISTS {tablename} CASCADE;")
        else:
            self.cur.execute(f"DROP TABLE IF EXISTS {tablename};")
        self.conn.commit()
        self.disconnect_databases()

    def save_feature_df_to_DB(self, df, tablename):
        df = df.dropna(subset = ["address"]) # is primary index, so we drop the one nan value, could also call it "nan" in string or sth
        df.columns = [col.lower() for col in df.columns] # its easier to work with lowercase column names, perhaps have to change later
        create_table = "CREATE TABLE IF NOT EXISTS " + tablename + " ("
        for col, dtype in zip(df.columns, df.dtypes):
            if col == "address":
                create_table += f"{col} TEXT, "
            elif dtype == "float64":
                create_table += f"{col} DOUBLE PRECISION, "
            elif dtype == "int64":
                create_table += f"{col} BIGINT, "
            elif dtype == "int32":
                create_table += f"{col} INTEGER, "
            elif dtype == "object":
                create_table += f"{col} NUMERIC(100,20), "  # for some reasons the max can be at least 25 digits, so we take 100 to be safe (then we get 100-20 digits)


            else:
                raise ValueError(f"Unknown dtype {dtype}")

        # make address column primary key, use replace
        create_table = create_table.replace("address TEXT", "address TEXT PRIMARY KEY")

        create_table = create_table[:-2] + ");"
        self.connect_databases()
        # delete if already exists
        self.cur.execute(create_table)

        self.conn.commit()
        df.to_sql(tablename, self.postgres_engine, method=psql_insert_df, if_exists="append", index=False)

        self.disconnect_databases()

    @log_time
    def process_piecewise(self, func, df_iter, args, data_prefix, tablename):

        # delete table if already exists
        self.delete_table_if_exists(tablename)

        def process_batch(i, batch_df, num_processes=4):
            time = datetime.now()
            self.logger.debug(f"Processing batch {i}")

            chunks = np.array_split(batch_df, num_processes)
            # reset index for each chunk
            chunks = [chunk.reset_index(drop=True) for chunk in chunks]

            if args == ():
                func_arguments = [[chunk] for chunk in chunks]
            else:

                func_arguments = [(chunk, *args) for chunk in chunks]

            if num_processes > 1:
                # Create a pool of processes
                pool = Pool(processes=num_processes)
                # Process the chunks in parallel
                results = pool.starmap(func, func_arguments)

                # Close the pool
                pool.close()
                pool.join()
            else: # mostly for debugging
                results = [func(*args_) for args_ in func_arguments]

            # Combine the results
            df = pd.concat(results)
            df = df.round(decimals=16)


            df.columns = [f"{data_prefix}__{col}" if col != "address" else col for col in df.columns]
            self.save_feature_df_to_DB(df, tablename)
            self.logger.debug(f"Saved batch {i} to DB in {datetime.now() - time}")

        atleast1iter = False
        for i, batch in enumerate(df_iter):
            atleast1iter = True
            process_batch(i, batch)

        if not atleast1iter:
            df = pd.DataFrame(columns=["address"])
            self.save_feature_df_to_DB(df, tablename)


        return tablename

    @log_time
    def tx_based_features(self) -> List[str]:
        """
        Features based on data directly in the transaction independent of the smart contract. block number, gas price...
        :return:
        """

        df_features_in = self.process_piecewise(tx_time_based_in_features, self.group_time_based_postgre_in(), (), "tx", "tx_time_based_in_features")
        df_features_out = self.process_piecewise(tx_time_based_out_features, self.group_time_based_postgre_out(), (), "tx", "tx_time_based_out_features")
        df_value_based = self.process_piecewise(tx_value_based_features, self.group_value_based_postgre(), (), "tx", "tx_value_based_features")
        df_custom_value = self.process_piecewise(tx_block_based_features, self.group_time_based_postgre_out(), (self.configs["Metadata"]["n_blocks"],), "tx", "tx_custom_features")

        generic_features = self.process_piecewise(tx_generic_features, self.group_tx_generic_postgre(), (self.configs["Aggregate"]["tx_generic_cols"], ), "tx", "tx_generic_features")

        df_folders = [df_features_in, df_features_out, df_value_based, generic_features, df_custom_value]

        return df_folders

    def get_swaps_info(self):
        function_signatures_with_index_path = f"{self.prefix}/data_lightweight/function_signatures_with_index.csv"
        multicall_function_signatures_path = f"{self.prefix}/data_lightweight/multicall_function_signatures_with_index.csv"
        df = pd.concat([pd.read_csv(function_signatures_with_index_path), pd.read_csv(multicall_function_signatures_path)])
        # get the rows that have the word "swap" in the text_signature column, caps shouldnt matter
        df = df[df["text_signature"].str.contains("swap", case=False)]
        # for each row get the colnames in the column col_names that contain "amount", caps shouldnt matter
        df["amount_cols"] = df["col_names"].apply(lambda x: [col for col in x.split(",") if "amount" in col.lower()])

        return df

    def get_events_info(self):
        events_path = f"{self.prefix}/data_lightweight/event_signatures_with_index.csv"
        df = pd.read_csv(events_path)
        # get the rows that have the word "swap" in the text_signature column, caps shouldnt matter
        df = df[df["text_signature"].str.contains("swap", case=False)]
        # for each row get the colnames in the column col_names that contain "amount", caps shouldnt matter
        df["amount_cols"] = df["col_names"].apply(lambda x: [col for col in x.split(",") if "amount" in col.lower()])

        return df

    @log_time
    def function_based_features(self) -> List[str]:

        df_generator_generator = self.group_swaps_postgre()
        tablenames = []
        while True:
            try:
                i, amount_cols, df_generator = next(df_generator_generator)
            except StopIteration:
                break

            swap_i_tablename = f"sb_{i}"
            swap_i_tablename = self.process_piecewise(swap_based_features, df_generator,
                                                      (amount_cols, i, self.configs["Metadata"]["n_blocks"]),
                                                      "scb__sb", swap_i_tablename)
            tablenames.append(swap_i_tablename)

        # create view that joins tablenames on address
        db_tablename = "swap_features"
        query = f"""
        DROP TABLE IF EXISTS {db_tablename};
        CREATE TABLE {db_tablename} AS
        SELECT * FROM {tablenames[0]}
        """
        for tablename in tablenames[1:]:
            query += f"""
            FULL OUTER JOIN {tablename} USING (address)
            """
        self.connect_databases()
        self.cur.execute(query)
        self.conn.commit()
        self.disconnect_databases()

        for tablename in tablenames:
            self.delete_table_if_exists(tablename)

        return [db_tablename]

    def merge_dfs_on_address(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        from functools import reduce
        return reduce(lambda left, right: pd.merge(left, right, on="address", how="outer"), dfs)

    def create_table_from_query(self, query,  tablename):
        createquery = f"CREATE TABLE {tablename} AS {query}"
        self.connect_databases()
        self.cur.execute(createquery)
        self.conn.commit()
        self.disconnect_databases()

    @log_time
    def merge_dfs_on_address_DB(self, tablenames: List[str]) -> None:
        """
        Join all the tables in tablenames on the address column in the postgre database
        :param dfs:
        :return:
        """
        # all the tables have a column address, so we can join on that
        # we want to join all the tables on the address column
        # we can get the addresses from the accounts table and left join on that
        column_names = []
        self.connect_databases()
        for tablename in tablenames:
            self.cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{tablename}'")
            columns = self.cur.fetchall()
            column_names += [f"{tablename}.{column[0]}" for column in columns if column[0] != "address"]

        index_creation_query = "\n".join(f"CREATE INDEX {tablename}_address_index ON {tablename} (address);" for tablename in tablenames)
        # Create the query string
        include_cols = ', '.join(column_names)
        query = f'SELECT accounts.address, {include_cols} FROM accounts LEFT JOIN {tablenames[0]} ON {tablenames[0]}.address = accounts.address'


        query += ' ' + ' '.join(f'LEFT JOIN {tablename} ON {tablename}.address = accounts.address' for tablename in tablenames[1:])
        self.logger.debug(index_creation_query)
        self.logger.debug(query)
        self.cur.execute(index_creation_query)
        self.conn.commit()
        self.disconnect_databases()


        # create new table with the output of the query
        self.delete_table_if_exists("features")
        self.create_table_from_query(query, "features")

        # delete the tables
        for tablename in tablenames:
            self.delete_table_if_exists(tablename)


    def get_addresses(self):
        """
        address is pk, so its unique
        :return:
        """
        query = """
            SELECT address FROM accounts
        """

        return self.get_df_generator(query)


    def group_swaps_postgre(self):

        df = self.get_swaps_info()
        # for each row in the df do the following:
        # Get the name of the table in column tablename. Build a query string that aggregates on the address column
        # get the array_agg of the columns in "amount_cols" and the length of the path column / count of the path column?
        # execute the query and save the result in a dataframe
        for i, row in df.iterrows():
            self.logger.debug(f"Grouping data for table {row['tablename']}")
            tablename = row["tablename"]
            amount_cols = row["amount_cols"]
            path_col = "path"

            querypart_amounts = ", ".join([f'array_agg(t."{col}") as {col}' for col in amount_cols])
            querypart_pathlength = f"array_agg(LENGTH({path_col}) - LENGTH(REPLACE({path_col}, ',', ''))) as pathlength"
            query = f'SELECT from_address as address, {querypart_amounts}, {querypart_pathlength} FROM public."{tablename}" as t GROUP BY from_address'
            yield i, amount_cols, self.get_df_generator(query)


    def group_erc20transfers_postgre(self):

        ### GROUP TOKEN TRANSFERS
        tablename = "transferErc20"
        self.logger.debug(f"Grouping data for table {tablename}")
        amount_cols = ["value"]

        querypart_amounts = ", ".join([f'array_agg(t."{col}") as {col}' for col in amount_cols])
        query = f'SELECT t.from as address, {querypart_amounts} FROM public."{tablename}" as t GROUP BY t.from'
        return self.get_df_generator(query)

    def group_swapevents_postgre(self):

        ### GROUP SWAPS
        tablename = "swap"
        self.logger.debug(f"Grouping data for table {tablename}")

        query = """
        SELECT address, array_agg(z.amountIn) as amountIn
        FROM (
            SELECT t.to as address, t."amount0In" + t."amount1In" as amountIn 
            FROM public."swap" as t
            UNION
            SELECT t.recipient as address, GREATEST(t."amount0", t."amount1") as amountIn
            FROM public."swapV3" as t
        ) as z
        GROUP BY address;
        """
        return self.get_df_generator(query)

    @log_time
    def event_based_features(self):

        erc20_table_name = self.process_piecewise(erc20_transfer_features, self.group_erc20transfers_postgre(),
                                                  (self.configs["Metadata"]["n_blocks"], ), "scb__eb__transfer",
                                                  "ev_erc20_based_features")

        swap_table_name = self.process_piecewise(event_swap_features, self.group_swapevents_postgre(),
                                                  (self.configs["Metadata"]["n_blocks"], ), "scb__eb__swap",
                                                  "ev_swap_based_features")

        return [erc20_table_name, swap_table_name]


    @log_time
    def ETL(self) -> None:
        """
        Combine all the features into one dataset

        There are address based features x
        There are transaction based features
            - time based custom features x
            - value based custom features
        There are smart contract based features

        For each attribute in our raw data, we have multiple possibilities to create features.
        For some, it is worth it to hand craft features, for others a base procedure can be applied

        The two base procedures are:
        1. Statistics: For large numbers or floats like "value". Displays the mean, median, std, min, max
        2. Distribution: For class data like "transaction_type". Displays what percentage of the transactions are
        of a certain type

        Custom procedures:
        1. Address based features: For example the number of starting zeros in the address
        2. Average distance between transactions in the time series
        ...

        """

        ### LOAD DATA
        address_based_tablenames = self.address_based_features()
        tx_based_tablenames = self.tx_based_features()
        sc_based_tablenames = self.function_based_features()
        event_based_tablenames = self.event_based_features()
        tables_to_join = address_based_tablenames + tx_based_tablenames + sc_based_tablenames + event_based_tablenames
        self.merge_dfs_on_address_DB(tables_to_join)


    @log_time
    def load_features(self, n_rows=None, preprocessing_function=None) -> pd.DataFrame:

        limitsnippet = f"LIMIT {n_rows}" if n_rows is not None else ""
        query = f"""
            SELECT * FROM features 
            where tx__custom__n_tx_per_block is not null -- not needed in test blocks and not needed in longitudinal study
            {limitsnippet}
        """

        # if we have a preprocessing function, only query in chunks of 1_000_000 and preprocess, then concat
        chunksize = 1_000_000

        cache_folder = self.configs["General"]["PREFIX_DB"] + "/cache"
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        cache_file = cache_folder + "/features.pkl"

        if os.path.exists(cache_file):
            self.logger.info("Loading features from cache")
            df_features = pd.read_pickle(cache_file)
        else:
            if preprocessing_function:

                self.connect_databases()
                self.cur.execute(query)

                column_names = [desc[0] for desc in self.cur.description]

                df_chunks = []
                while True:
                    rows = self.cur.fetchmany(chunksize)
                    if not rows:
                        break
                    df = pd.DataFrame(rows, columns=column_names)
                    df_chunk = preprocessing_function(df)
                    df_chunks.append(df_chunk)

                df_features = pd.concat(df_chunks)

                self.disconnect_databases()
            else:
                df_features = self.get_df(query)

            df_features.index = df_features["address"]
            df_features.drop("address", axis=1, inplace=True)
            df_features.to_pickle(cache_file)

        return df_features

    @log_time
    def load_transfers(self, n_rows=None, cache=False):
        """
        There are transferErc20 and transferEth tables. load both
        from to in transferErc20
        from_address to_address in transferEth

        Remove self transfers and remove contract addresses. Only EOAs
        :return:
        """

        import os
        cache_file = f"{self.prefix}/data/transfer_cache.parquet"
        if os.path.exists(cache_file) and cache:
            df = pd.read_parquet(cache_file)
            return df


        limit_str = f"LIMIT {int(n_rows/2)}" if n_rows else ""
        query = f"""
            SELECT * from (
            SELECT erc.from as from_address, erc.to as to_address FROM public."transferErc20" as erc {limit_str}
                        UNION
                        SELECT tf.from_address, tf.to_address FROM public."transferEth" as tf {limit_str}
            ) as transactions
            
            WHERE transactions.from_address IN (
                SELECT address
                FROM accounts
                WHERE code IS NULL
            )
            AND transactions.to_address IN (
                SELECT address
                FROM accounts
                WHERE code IS NULL
            )
            AND transactions.to_address != transactions.from_address
;

        """

        df = self.get_df(query)
        # drop row with None value in index
        df = df.dropna(subset=["from_address", "to_address"])
        if cache:
            df.to_parquet(cache_file)

        return df


    @log_time
    def load_features_test(self):
        query = """
                SELECT ft.*
        FROM (
            SELECT DISTINCT from_address
            FROM public.transactions
        ) AS distinct_addresses
        JOIN (
            SELECT from_address
            FROM public.transactions
            where block_id <= 15599999
            ORDER BY block_id DESC
            LIMIT 100000
        ) AS last_blocks ON distinct_addresses.from_address = last_blocks.from_address
        JOIN public.features ft ON last_blocks.from_address = ft.address;

        """
        df = self.get_df(query)

        df.index = df["address"]
        df.drop("address", axis=1, inplace=True)

        return df


    def load_MEVinspect_transactions(self, type):
        """
        For reproducibility reasons we only consider MEV inspect results
        bit long of a query, but fast enough
        :return:
        """

        assert type in ["arbitrages", "sandwiches", "liquidations"]

        datapath = f"{self.prefix}/data/mev_inspect_predictions/{type}.csv"
        # has columns block_number,tx_hash
        df = pd.read_csv(datapath)
        # get all unique tx_
        # query all txs from db
        self.logger.debug("Loading arbitrage txs from db")
        txs = df["tx_hash"].unique()
        txs_str = "( '" + "', '".join(txs) + "' )"

        query = f"""
            SELECT * FROM public.transactions t
            WHERE t.tx_hash IN {txs_str}
        """
        df_txs = self.get_df(query)
        return df_txs
    def load_MEVinspect(self, type):
        """
        For reproducibility reasons we only consider MEV inspect results
        bit long of a query, but fast enough
        :return:
        """

        df_txs = self.load_MEVinspect_transactions(type)
        unique_from_addresses = df_txs["from_address"].unique()

        # query all features from db
        addresses_str = "( '" + "', '".join(unique_from_addresses) + "' )"
        query = f"""
            SELECT * FROM public.features f
            WHERE f.address IN {addresses_str}
        """

        df_features = self.get_df(query)
        df_features.index = df_features["address"]
        df_features.drop("address", axis=1, inplace=True)
        return df_features

    def load_unsupervised_features(self, preprocessing_function=None):
        # load txt from data/walltes_to_annotate.csv
        with open(f"{self.prefix}/data/wallets_to_annotate.csv") as f:
            addresses = f.read()

        addresses_split = addresses.lower().split("\n")
        addresses_split = [a for a in addresses_split if a != ""]
        t = "( '" + "', '".join(addresses_split) + "' )"

        query = f"""

                SELECT * FROM public.features f 
                --where f.address NOT IN
                --{t}
                """


        # if we have a preprocessing function, only query in chunks of 1_000_000 and preprocess, then concat
        chunksize = 1_000_000
        cache_folder = self.configs["General"]["PREFIX_DB"] + "/cache"
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        cache_file = cache_folder + "/features_unsupervised.pkl"

        if os.path.exists(cache_file):
            self.logger.info("Loading features from cache")
            df = pd.read_pickle(cache_file)
        else:
            if preprocessing_function:

                self.connect_databases()
                self.cur.execute(query)

                column_names = [desc[0] for desc in self.cur.description]

                df_chunks = []
                while True:
                    rows = self.cur.fetchmany(chunksize)
                    if not rows:
                        break
                    df = pd.DataFrame(rows, columns=column_names)
                    df_chunk = preprocessing_function(df)
                    df_chunks.append(df_chunk)

                df = pd.concat(df_chunks)

                self.disconnect_databases()
            else:

                df = self.get_df(query)


            df.index = df["address"]
            df.drop("address", axis=1, inplace=True)
            df.to_pickle(cache_file)



        return df


    def load_evalset_features(self):
        # load txt from data/walltes_to_annotate.csv
        with open(f"{self.prefix}/data/wallets_to_annotate.csv") as f:
            addresses = f.read()

        addresses_split = addresses.lower().split("\n")
        addresses_split = [a for a in addresses_split if a != ""]
        t = "( '"+ "', '".join(addresses_split) + "' )"

        query = f"""
            
                SELECT * FROM public.features f 
                where f.address IN
                {t}
                """

        df = self.get_df(query)
        df.index = df["address"]
        df.drop("address", axis=1, inplace=True)
        return df

if __name__ == "__main__":
    configs = load_configs("..")
    p = Aggregate(configs, "..")
    p.ETL()
