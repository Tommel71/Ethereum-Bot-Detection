from tools import load_configs, load_json, load_mapping
from tools import get_filehandler
import os
from tqdm import tqdm
import logging
import psycopg2
import time
from src.Preprocessing.handle_functions import Decoder
import sqlalchemy
from sqlalchemy import text
from collections import defaultdict
import json
import pandas as pd
from tools import psql_insert_df

dtype_mapping = {
    "address": "TEXT",
    "uint256": "NUMERIC(78,0)",
    "uint160": "NUMERIC(78,0)",
    "uint128": "NUMERIC(78,0)",
    "int24": "INT",
    "uint8": "INT",
    "uint32": "INT",
    "uint64": "NUMERIC(30,0)",
    "int256": "NUMERIC(78,0)",
    "address[]": "TEXT",
    "bytes": "TEXT",
    "bytes[]" : "TEXT"
}


def load_table_settings(prefix, file):

    path = f"{prefix}/data_lightweight/{file}"
    df = pd.read_csv(path)

    def handle_json(x):
        if pd.isna(x):
            x = "{}"
        x = x.replace("'", '"')
        return json.loads(x)


    table_settings = {}
    for _, row in df.iterrows():
        name = row["tablename"]
        fk_setting_string = row["foreignkeys"]
        added_columns = row["added_columns"]

        table_settings[name] = {"foreign_key_settings": handle_json(fk_setting_string),
                                "hash": row["hex_signature_hash"],
                                "added_columns": handle_json(added_columns)}


    return table_settings


class Preprocess:

    def __init__(self, configs, prefix):
        self.configs = configs
        self.prefix = prefix
        self.prefix_db = configs["General"]["PREFIX_DB"]
        self.mapping_events = load_mapping(prefix + "/data/event_signatures.csv")
        self.mapping_functions = load_mapping(prefix + "/data/signatures.csv")

        # create logger that saves to prefix/logs
        self.logger = logging.getLogger("preprocessing")
        self.logger.setLevel(logging.DEBUG)
        fh = get_filehandler(prefix, "preprocessing")
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(fh)
        self.postgres_credentials = load_json(prefix + f"/credentials/"
                                                       f"postgres/{self.configs['General']['run_name']}.json")



    def connect_databases(self):

        self.conn = psycopg2.connect(host=self.postgres_credentials["host"], port=self.postgres_credentials["port"],
                                database=self.postgres_credentials["database"], user=self.postgres_credentials["user"],
                                password=self.postgres_credentials["password"])
        # create sqlalchemy engine
        self.postgres_engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{self.postgres_credentials['user']}:{self.postgres_credentials['password']}@"
            f"{self.postgres_credentials['host']}:{self.postgres_credentials['port']}"
            f"/{self.postgres_credentials['database']}", execution_options=dict(stream_results=True))

        self.cur = self.conn.cursor()

        with self.conn.cursor(name='iter_cursor') as cursor:
            cursor.itersize = 100000  # chunk size



    def disconnect_databases(self):


        self.cur.close()
        self.conn.close()

    def postgres_add_transactions(self):

        pks = ["block_id", "transaction_index"] # TODO just changed that from tx_hash. check if this slows down anything

        bigints = ["nonce", "gas_price", "block_timestamp", "receipt_effective_gas_price"]
        ints = ['transaction_index', 'gas', "transaction_type",
                'receipt_cumulative_gas_used', 'receipt_gas_used', 'receipt_status',
                'block_id']
        numerics = ['value', 'max_fee_per_gas', 'max_priority_fee_per_gas', 'transaction_index_relative']

        columns = ['nonce', 'transaction_index', 'from_address', 'to_address', 'value',
                'gas', 'gas_price', 'input', 'block_timestamp', 'block_hash',
                'max_fee_per_gas', 'max_priority_fee_per_gas', 'transaction_type',
                'receipt_cumulative_gas_used', 'receipt_gas_used',
                'receipt_contract_address', 'receipt_root', 'receipt_status',
                'receipt_effective_gas_price', 'tx_hash', 'tx_hash_prefix', 'block_id', 'transaction_index_relative']

        base_dir = self.prefix_db + "/erigon_extract/uncompressed/transactions/"
        table_name = "transactions"
        self.add_csvs_to_postgres(base_dir, columns, ints, numerics, bigints, pks, table_name)

    def postgres_add_blocks(self):
        columns = ['parent_hash', 'nonce', 'sha3_uncles', 'logs_bloom',
       'transactions_root', 'state_root', 'receipts_root', 'miner',
       'difficulty', 'total_difficulty', 'size', 'extra_data', 'gas_limit',
       'gas_used', 'timestamp', 'transaction_count', 'base_fee_per_gas',
       'block_id', 'block_id_group', 'block_hash']

        ints = ["size", "transaction_count", "block_id_group"]
        bigints = ["gas_limit", "gas_used", "timestamp", "block_id"]
        numerics = ["difficulty, total_difficulty, base_fee_per_gas"]

        pks = ["block_id"]
        base_dir = self.prefix_db + "/erigon_extract/uncompressed/blocks/"
        table_name = "blocks"
        self.add_csvs_to_postgres(base_dir, columns, ints, numerics, bigints, pks, table_name)

    def postgres_add_codes(self):

        base_dir = self.prefix_db + "/erigon_extract/uncompressed/codes/"
        if self.configs["General"]["run_name"] == "dev":
            base_dir = self.prefix_db + "/erigon_extract/uncompressed/codes/"

        table_name = "codes"
        # if the csv doesnt have 5 column names, add 'code_id' as the first name

        columns = ['code_id', 'block_number', 'from_address', 'to_address', 'output']
        ints = ["code_id"]
        bigints = []
        numerics = []
        pks = ["code_id"]
        self.add_csvs_to_postgres(base_dir, columns, ints, numerics, bigints, pks, table_name)
        self.connect_databases()



    def postgres_add_logs(self):

        base_dir = self.prefix_db + "/erigon_extract/uncompressed/logs/"
        info = ['log_index', 'transaction_index', 'block_hash', 'address',
       'transaction_hash', 'data', 'topics', 'tx_hash', 'block_id',
       'block_id_group']

        ints = ["log_index", "transaction_index", "block_id", "block_id_group"]
        numerics = []
        bigints = []
        pks = ["block_id", "log_index"]
        table_name = "logs"

        self.add_csvs_to_postgres(base_dir, info, ints, numerics, bigints, pks, table_name)

    def add_csvs_to_postgres(self, base_dir, columns, ints, numerics, bigints, pks, table_name):
        self.logger.debug(f"Add {table_name} to postgres")

        csvs = os.listdir(base_dir)
        csvs = [csv for csv in csvs if csv.endswith(".csv")]

        # create table
        query_create = f''' DROP TABLE IF EXISTS {table_name} CASCADE;
                    CREATE TABLE IF NOT EXISTS {table_name} (
                    {"".join([f"{columns[i]} text, " for i in range(len(columns))])}
                    PRIMARY KEY ({"".join([f"{pks[i]}, " for i in range(len(pks))])[0:-2]})
                    );'''

        for int_ in ints:
            query_create = query_create.replace(f" {int_} text", f" {int_} integer")

        for numeric in numerics:
            query_create = query_create.replace(f" {numeric} text", f" {numeric} NUMERIC(25)")

        for bigint in bigints:
            query_create = query_create.replace(f" {bigint} text", f" {bigint} bigint")

        self.connect_databases()
        self.cur.execute(query_create)
        self.conn.commit()

        delimitersnippet = ""

        for csv_file in tqdm(csvs):

            csv_path = base_dir + csv_file
            query_populate = f""" COPY {table_name}({''.join([f'{columns[i]}, ' for i in range(len(columns))])[0:-2]})
                        FROM '{csv_path}'
                        WITH (FORMAT csv{delimitersnippet}, HEADER true);"""

            self.cur.execute(query_populate)
            self.conn.commit()

        self.disconnect_databases()

        self.logger.debug(f"Added {table_name} to postgres")

    def add_indices(self):
        self.logger.debug("Add indices")
        self.connect_databases()
        self.cur.execute("CREATE INDEX IF NOT EXISTS tx_idx_tx_hash ON transactions (tx_hash);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS tx_idx_tx_from ON transactions (from_address);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS tx_idx_tx_to ON transactions (to_address);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS tx_idx_block_id ON transactions (block_id);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS tx_tx_idx ON transactions (transaction_index);")

        self.cur.execute("CREATE INDEX IF NOT EXISTS blocks_idx_block_id ON blocks (block_id);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS logs_idx_block_id ON logs (block_id);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS codes_idx_to_address ON codes (to_address)")
        self.conn.commit()
        self.disconnect_databases()
        self.logger.debug("Added indices")

    def add_foreign_keys(self, event_tables_settings, function_tables_settings):
        self.logger.debug("Add foreign keys")
        start = time.time()
        self.connect_databases()
        self.cur.execute("ALTER TABLE transactions ADD FOREIGN KEY (block_id) REFERENCES blocks(block_id);")
        self.cur.execute("ALTER TABLE transactions ADD FOREIGN KEY (from_address) REFERENCES accounts(address);")
        self.cur.execute("ALTER TABLE transactions ADD FOREIGN KEY (to_address) REFERENCES accounts(address);")
        self.cur.execute("ALTER TABLE logs ADD FOREIGN KEY (block_id, transaction_index) REFERENCES transactions;")

        for name in event_tables_settings.keys():
            self.logger.debug(f"Add foreign key for {name}")
            query = ""
            for fk_setting in event_tables_settings[name]["foreign_key_settings"]:
                query += f'''
                ALTER TABLE "{name}" ADD FOREIGN KEY ("{fk_setting["column"]}") 
                REFERENCES "{fk_setting["table_foreign"]}"("{fk_setting["column_foreign"]}");
                '''
            self.cur.execute(query)

        for name in function_tables_settings.keys():
            self.logger.debug(f"Add foreign key for {name}")
            query = ""
            for fk_setting in function_tables_settings[name]["foreign_key_settings"]:
                query += f'''
                ALTER TABLE "{name}" ADD FOREIGN KEY ("{fk_setting["column"]}") 
                REFERENCES "{fk_setting["table_foreign"]}"("{fk_setting["column_foreign"]}");
                '''

            if len(query)>0:
                self.cur.execute(query)


        self.conn.commit()
        self.disconnect_databases()
        self.logger.debug(f"Added foreign keys in {time.time() - start} seconds")

    def add_delete_cascade(self):
        self.logger.debug("Add delete cascade")
        start = time.time()
        self.connect_databases()
        self.cur.execute("ALTER TABLE logs ADD FOREIGN KEY (tx_hash) REFERENCES transactions(tx_hash) ON DELETE CASCADE;")
        self.conn.commit()
        self.disconnect_databases()
        self.logger.debug(f"Added delete cascade in {time.time() - start} seconds")

    def drop_anomalous(self):
        """
        Some blocks are not in the traces, so we drop them
        :return:
        """
        #list_anomalous_block_ids = [15539509]
        self.logger.debug(f"Drop anomalous accs")
        self.connect_databases()
        # get to and from addresses from transactions and remove them from accounts if they re in blocks

        query = """
        DELETE FROM transactions
            WHERE from_address IN
            (
                SELECT distinct b.from_address as r_address
                FROM ( SELECT address, t.from_address FROM transactions t LEFT JOIN accounts a ON a.address = t.from_address) b
                WHERE b.address IS NULL
            );
            
        """
        self.cur.execute(query)
        self.logger.debug(f"Postgres: {self.cur.statusmessage}")

        query = """
        DELETE FROM transactions
            WHERE to_address IN
            (
                SELECT distinct b.to_address as r_address
                FROM ( SELECT address, t.to_address FROM transactions t LEFT JOIN accounts a ON a.address = t.to_address) b
                WHERE b.address IS NULL
            );

        """
        self.cur.execute(query)
        self.logger.debug(f"Postgres: {self.cur.statusmessage}")

        self.conn.commit()
        self.disconnect_databases()
        self.logger.debug(f"Dropped anomalous txes")

    def create_event_tables(self, event_tables_settings):
        """
        Work out topics, function names, etc.
        :return:
        """
        # query logs from postgres
        self.logger.debug(f"Logs processing")
        self.connect_databases()

        decoder = Decoder(prefix=self.prefix)

        signature_hashes = {x["hash"] for x in event_tables_settings.values()}
        signature_hash_to_name = {event_tables_settings[x]["hash"]: x for x in event_tables_settings.keys()}

        for name in event_tables_settings.keys():
            types, names = decoder.get_params_names(event_tables_settings[name]["hash"])

            types_postgres = [dtype_mapping[t] for t in types]

            event_tables_settings[name].update({"columns":names, "types": types_postgres})
            snippet = ", ".join([f'"{x}" {y}' for x, y in zip(names, types_postgres)]) + " ,"
            if snippet == " ,":
                snippet = ""
            query = f''' 
            DROP TABLE IF EXISTS "{name}";
            CREATE TABLE IF NOT EXISTS "{name}" (
                {snippet}
                block_id INT,
                log_index INT,
                PRIMARY KEY (block_id, log_index)
            );
                
            '''
            self.cur.execute(query)

        query = f'''
        
        DROP TABLE IF EXISTS "logsInfo";
            CREATE TABLE IF NOT EXISTS "logsInfo" (
                function_name TEXT,
                signature_hash TEXT,
                param_types TEXT,
                param_values TEXT,
                block_id INT,
                log_index INT,
                PRIMARY KEY (block_id, log_index)
            );
                
        
        '''
        self.cur.execute(query)

        self.conn.commit()


        query = "SELECT topics, data, block_id, log_index FROM logs;"

        self.logger.debug(f"Postgres: {self.cur.statusmessage}")

        chunksize = 100_000
        with self.postgres_engine.connect() as conn:

            for chunk in pd.read_sql(text(query), conn, chunksize=chunksize):

                # faster:
                data_tables = defaultdict(list)
                infos = chunk.apply(lambda x: {**decoder.decode_log(x["topics"], x["data"]), "block_id": x["block_id"], "log_index": x["log_index"]}, axis=1)
                infos.apply(lambda x: data_tables[signature_hash_to_name[x["signature_hash"]]].append(x) if x["signature_hash"] in signature_hashes and not (len(x["param_types"]) == 1 and x["param_types"][0] == "decoding_unsuccessful_flag") else None)
                logsInfo = pd.DataFrame.from_records(infos)

                # change columns param_types and param_values to text
                logsInfo["param_types"] = logsInfo["param_types"].apply(lambda x: "|".join(x))

                logsInfo.to_sql("logsInfo", self.postgres_engine, method=psql_insert_df, if_exists="append", index=False)
                # persist data tables
                for key in data_tables.keys():
                    series = pd.DataFrame(data_tables[key])["param_values"]

                    # split list of values into columns
                    df = pd.DataFrame(series.values.tolist(), index=series.index)
                    df.columns = event_tables_settings[key]["columns"]
                    df["block_id"] = pd.DataFrame(data_tables[key])["block_id"]
                    df["log_index"] = pd.DataFrame(data_tables[key])["log_index"]
                    df.to_sql(key, self.postgres_engine, method=psql_insert_df, if_exists="append", index=False)

                self.logger.debug(f"Processed {chunksize} logs")

            self.logger.debug(f"Logs processed")

            ### ADD FOREIGN KEYS TO NEW TABLES

            # for all tables, set up foreign keys
            for name in event_tables_settings.keys():
                query = f'ALTER TABLE "{name}" ADD FOREIGN KEY (block_id, log_index) REFERENCES logs;'
                self.cur.execute(query)

            query = f'ALTER TABLE "logsInfo" ADD FOREIGN KEY (block_id, log_index) REFERENCES logs;'
            self.cur.execute(query)
            self.conn.commit()


        self.disconnect_databases()

        self.logger.debug(f"Indices created")


    def create_multicall_tables(self, tx_tables_settings):
        # query logs from postgres
        self.logger.debug(f"Creating multicall tables")
        self.connect_databases()

        decoder = Decoder(prefix=self.prefix)

        signature_hashes = {x["hash"] for x in tx_tables_settings.values()}
        signature_hash_to_name = {tx_tables_settings[x]["hash"]: x for x in tx_tables_settings.keys()}

        for name in tx_tables_settings.keys():
            types, names = decoder.get_params_names(tx_tables_settings[name]["hash"])
            types_postgres = [dtype_mapping[t] for t in types]

            tx_tables_settings[name].update({"columns": names, "types": types_postgres})
            snippet = ", ".join([f'"{x}" {y}' for x, y in zip(names, types_postgres)]) + " ,"
            if snippet == " ,":
                snippet = ""
            query = f''' 
                   DROP TABLE IF EXISTS "multicall_{name}";
                   CREATE TABLE IF NOT EXISTS "multicall_{name}" (
                       {snippet}
                       block_id INT,
                       transaction_index INT,
                       subtransaction_index INT,
                       PRIMARY KEY (block_id, transaction_index, subtransaction_index)
                   );

                   '''
            self.cur.execute(query)

        query = f'''

               DROP TABLE IF EXISTS "multicall_functionInfo";
                   CREATE TABLE IF NOT EXISTS "multicall_functionInfo" (
                       function_name TEXT,
                       signature_hash TEXT,
                       param_types TEXT,
                       param_values TEXT,
                       block_id INT,
                       transaction_index INT,
                       subtransaction_index INT,
                       PRIMARY KEY (block_id, transaction_index, subtransaction_index)
                   );


               '''
        self.cur.execute(query)

        self.conn.commit()

        query = "SELECT input, block_id, transaction_index FROM transactions WHERE substring(input FROM 1 FOR 10) = '0x5ae401dc';"

        self.logger.debug(f"Postgres: {self.cur.statusmessage}")

        chunksize = 100_000
        with self.postgres_engine.connect() as conn:

            for chunk in pd.read_sql(text(query), conn, chunksize=chunksize):
                if len(chunk) == 0:
                    continue

                data_tables = defaultdict(list)
                infos_raw = chunk.apply(lambda x: (decoder.decode_function_multi(x["input"]), x["block_id"], x["transaction_index"]), axis=1)
                infos_raw.apply(lambda x: [x[0][i].update({"block_id": x[1], "transaction_index": x[2], "subtransaction_index": i}) for i in range(len(x[0]))])
                infos = pd.concat([pd.Series(x[0]) for x in infos_raw], ignore_index=True)

                infos.apply(lambda x: data_tables[signature_hash_to_name[x["signature_hash"]]].append(x) if x["signature_hash"] in signature_hashes and not (
                            len(x["param_types"]) == 1 and x["param_types"][
                        0] in ["decoding_unsuccessful_flag", "decoding_unsuccessful_flag_mc"]) else None)

                functionInfo = pd.DataFrame.from_records(infos)



                # TODO this is an unclean way of solving the issue. Correct way would be
                # for the function and multicall decoding to also insert the functions/multicalls that failed to decode
                # right now they are filtered above with the == "decoding_unsuccessful_flag" condition
                # as soon as this is done, one would have to adapt the feature engineering part aswell.
                # another problem is that we cannot pass a text flag in the parameter columsn because of types
                # we would need an extra column and set the other columns to null. (maybe just null works too but would need thinking)
                functionInfo= functionInfo[[x != ["decoding_unsuccessful_flag_mc"] for x in functionInfo["param_types"]]]

                #dup = functionInfo[functionInfo.duplicated(subset=["block_id", "transaction_index", "subtransaction_index"], keep=False)]

                # change columns param_types and param_values to text
                functionInfo["param_types"] = functionInfo["param_types"].apply(lambda x: "|".join(x))

                functionInfo.to_sql('multicall_functionInfo', con=self.postgres_engine, method=psql_insert_df, if_exists="append",
                                    index=False)

                # persist data tables
                for key in data_tables.keys():
                    series = pd.DataFrame(data_tables[key])["param_values"]

                    # split list of values into columns
                    df = pd.DataFrame(series.values.tolist(), index=series.index)

                    df.columns = tx_tables_settings[key]["columns"]
                    df["block_id"] = pd.DataFrame(data_tables[key])["block_id"]
                    df["transaction_index"] = pd.DataFrame(data_tables[key])["transaction_index"]
                    df["subtransaction_index"] = pd.DataFrame(data_tables[key])["subtransaction_index"]
                    # df.to_csv(f"debug/{key}.csv")
                    for col in df.columns:
                        if df[col].dtype == "uint64":
                            df[col] = df[col].astype("object")
                    df.to_sql("multicall_" + key, con=self.postgres_engine, method=psql_insert_df, if_exists="append", index=False)

                self.logger.debug(f"Processed {chunksize} transactions")

            self.logger.debug(f"Transactions processed")

            ### ADD FOREIGN KEYS TO NEW TABLES

            # for all tables, set up foreign keys
            for name in tx_tables_settings.keys():
                query = f'ALTER TABLE "multicall_{name}" ADD FOREIGN KEY (block_id, transaction_index) REFERENCES multicall;'
                self.cur.execute(query)

            query = f'ALTER TABLE "multicall_functionInfo" ADD FOREIGN KEY (block_id, transaction_index) REFERENCES multicall;'
            self.cur.execute(query)
            self.conn.commit()

        self.disconnect_databases()

        self.logger.debug(f"Indices created")


    def create_function_tables(self, tx_tables_settings):
        """
        Work out topics, function names, etc.
        :return:
        """
        # query logs from postgres
        self.logger.debug(f"Creating function tables")
        self.connect_databases()

        decoder = Decoder(prefix=self.prefix)

        signature_hashes = {x["hash"] for x in tx_tables_settings.values()}
        signature_hash_to_name = {tx_tables_settings[x]["hash"]: x for x in tx_tables_settings.keys()}

        for name in tx_tables_settings.keys():
            types, names = decoder.get_params_names(tx_tables_settings[name]["hash"])
            types_postgres = [dtype_mapping[t] for t in types]

            tx_tables_settings[name].update({"columns": names, "types": types_postgres})
            snippet = ", ".join([f'"{x}" {y}' for x, y in zip(names, types_postgres)]) + " ,"
            if snippet == " ,":
                snippet = ""
            query = f''' 
            DROP TABLE IF EXISTS "{name}";
            CREATE TABLE IF NOT EXISTS "{name}" (
                {snippet}
                block_id INT,
                transaction_index INT,
                PRIMARY KEY (block_id, transaction_index)
            );

            '''
            self.cur.execute(query)

        query = f'''

        DROP TABLE IF EXISTS "functionInfo";
            CREATE TABLE IF NOT EXISTS "functionInfo" (
                function_name TEXT,
                signature_hash TEXT,
                param_types TEXT,
                param_values TEXT,
                block_id INT,
                transaction_index INT,
                PRIMARY KEY (block_id, transaction_index)
            );


        '''
        self.cur.execute(query)

        self.conn.commit()

        query = "SELECT value, input, block_id, transaction_index FROM transactions;"

        self.logger.debug(f"Postgres: {self.cur.statusmessage}")

        chunksize = 100_000
        with self.postgres_engine.connect() as conn:

            for chunk in pd.read_sql(text(query), conn, chunksize=chunksize):

                data_tables = defaultdict(list)
                infos = chunk.apply(lambda x: {**decoder.decode_input_single(x["input"]), "block_id": x["block_id"], "transaction_index": x["transaction_index"]}, axis=1)

                #infos = chunk.apply(
                #    lambda x: [{**info.copy(), "block_id": x["block_id"], "transaction_index": x["transaction_index"]}
                #               for info in decoder.decode_input(x["input"])], axis=1)


                infos.apply(lambda x: data_tables[signature_hash_to_name[x["signature_hash"]]].append(x) if x["signature_hash"] in signature_hashes and not (len(x["param_types"]) == 1 and x["param_types"][0] == "decoding_unsuccessful_flag") else None)

                functionInfo = pd.DataFrame.from_records(infos)
                # change columns param_types and param_values to text
                functionInfo["param_types"] = functionInfo["param_types"].apply(lambda x: "|".join(x))

                functionInfo.to_sql('functionInfo', con=self.postgres_engine, method=psql_insert_df, if_exists="append", index=False)

                # persist data tables
                for key in data_tables.keys():
                    series = pd.DataFrame(data_tables[key])["param_values"]

                    # split list of values into columns
                    df = pd.DataFrame(series.values.tolist(), index=series.index)
                    df.columns = tx_tables_settings[key]["columns"]
                    df["block_id"] = pd.DataFrame(data_tables[key])["block_id"]
                    df["transaction_index"] = pd.DataFrame(data_tables[key])["transaction_index"]
                    #df.to_csv(f"debug/{key}.csv")
                    for col in df.columns:
                        if df[col].dtype == "uint64":
                            df[col] = df[col].astype("object")
                    df.to_sql(key, con=self.postgres_engine, method=psql_insert_df, if_exists="append", index=False)

                self.logger.debug(f"Processed {chunksize} transactions")

            self.logger.debug(f"Transactions processed")

            ### ADD FOREIGN KEYS TO NEW TABLES

            # for all tables, set up foreign keys
            for name in tx_tables_settings.keys():
                query = f'ALTER TABLE "{name}" ADD FOREIGN KEY (block_id, transaction_index) REFERENCES transactions;'
                self.cur.execute(query)

            query = f'ALTER TABLE "functionInfo" ADD FOREIGN KEY (block_id, transaction_index) REFERENCES transactions;'
            self.cur.execute(query)
            self.conn.commit()

        self.disconnect_databases()

        self.logger.debug(f"Indices created")

    def create_accounts(self, event_tables_settings, function_tables_settings):
        """ TODO MAKE THIS A PL PGSQL FUNCTION OTHERWISE ITS POSSIBLE THAT accountstemp remains in the DB
        Get the addresses used in transactions in to and from, and from the events that reference addresses.
        Then join sets to get all addresses and create accounts table.
        :param event_tables_settings:
        :return:
        """
        self.connect_databases()
        self.logger.debug("Creating accounts table")
        # create accounts table
        query = """
        DROP TABLE IF EXISTS accountstemp;
        CREATE TABLE IF NOT EXISTS accountstemp (
            address TEXT,
            PRIMARY KEY (address)
        );
        """
        self.cur.execute(query)
        self.conn.commit()

        # get all addresses from transactions
        query = """ INSERT INTO accountstemp(address)
        
        SELECT DISTINCT from_address FROM transactions
        WHERE from_address IS NOT NULL
        UNION
        SELECT DISTINCT to_address FROM transactions
        WHERE to_address IS NOT NULL
        """

        # get all addresses from events
        for name in event_tables_settings.keys():
            foreign_key_settings = event_tables_settings[name]["foreign_key_settings"]
            for foreign_key_setting in foreign_key_settings:
                if (foreign_key_setting["table_foreign"] == "accounts") and (foreign_key_setting["column_foreign"] == "address"):
                    query += f'''
                    UNION
                    SELECT DISTINCT "{foreign_key_setting["column"]}" FROM "{name}"
                    where "{foreign_key_setting["column"]}" is not null
                    '''

        for name in function_tables_settings.keys():
            foreign_key_settings = function_tables_settings[name]["foreign_key_settings"]
            for foreign_key_setting in foreign_key_settings:
                if (foreign_key_setting["table_foreign"] == "accounts") and (foreign_key_setting["column_foreign"] == "address"):
                    query += f'''
                    UNION
                    SELECT DISTINCT "{foreign_key_setting["column"]}" FROM "{name}"
                    where "{foreign_key_setting["column"]}" is not null
                    '''


        self.cur.execute(query)
        self.conn.commit()

        # to make left join fast
        query_create_index = """
        CREATE INDEX IF NOT EXISTS idx_address ON accountstemp(address);
        """
        self.cur.execute(query_create_index)
        self.conn.commit()

        # Create accounts table, drop duplicates in the codes table after left joining with the temporary accounts table
        query = """
        DROP TABLE IF EXISTS accounts;
        CREATE TABLE IF NOT EXISTS accounts (
            address TEXT,
            code TEXT,
            account_type TEXT,
            PRIMARY KEY (address)
        );
        INSERT INTO accounts
        SELECT
                        i.address as address
                      , i.code as code
                      ,  (case when i.code is null then 'EOA' else 'CA' end)
                      
                      FROM (
                            SELECT
                                    address
                                  , code_id
                                , code
                                
                                  , ROW_NUMBER() OVER(PARTITION BY address ORDER BY code_id DESC) as rn
                            FROM (
                                SELECT address, code_id, output as code
                                FROM accountstemp
                                LEFT JOIN codes
                                ON accountstemp.address = codes.to_address
                            ) t
                      ) i
                        where 1=1
                        and i.rn = 1
                        or i.code_id IS NULL
        ;
        DROP TABLE IF EXISTS accountstemp;
        """
        
        self.cur.execute(query)
        # add index to address column
        self.cur.execute("CREATE INDEX accounts_address_idx on accounts (address);")
        self.conn.commit()

        self.disconnect_databases()
        self.logger.debug(f"Accounts created")

    def get_type(self, table, column):
        """
        Get the type of a column in a table
        :param table:
        :param column:
        :return:
        """
        query = f"SELECT data_type, numeric_precision FROM information_schema.columns WHERE table_name = '{table}' and column_name = '{column}'"
        self.cur.execute(query)
        a = self.cur.fetchone()
        if a[1] is not None:
            return f"{a[0]}({a[1]})"
        return a[0]


    def improve_function_tables(self, function_tables_settings, multicall=False):
        """
        Improve transferEth table by adding to_address and from_address and value
        :return:
        """
        self.connect_databases()
        self.logger.debug("Improving function tables")
        # left join with transactions to get to_address from_address ad value from transactions

        if multicall:
            table_prefix = "multicall_"
        else:
            table_prefix = ""


        fks = ["block_id", "transaction_index"]
        for name in function_tables_settings.keys():
            added_columns = function_tables_settings[name]["added_columns"]
            for foreign_table_name in added_columns.keys():
                cols = added_columns[foreign_table_name]
                query = ""
                name = f"{table_prefix}{name}"
                for col in cols:
                    query += f'''
                        DO $$
                        BEGIN
                        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = \'{name}\' AND COLUMN_NAME = \'{col}\') THEN
                            ALTER TABLE "{name}" ADD COLUMN "{col}" {self.get_type(foreign_table_name, col)};
                        END IF;
                        END$$;
                        '''

                # get all columns from table name
                query_get_columns = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{name}'"
                self.cur.execute(query_get_columns)
                columns_present = [x[0] for x in self.cur.fetchall()]
                # remove cols from columns_present
                for col in cols:
                    if col in columns_present:
                        columns_present.remove(col)

                # turn them into a string I can add to the query, and if the
                select_snippet1 = ", ".join([f'"{name}"."{x}"' for x in columns_present])
                select_snippet2 = ", ".join([f'"{foreign_table_name}"."{col}"' for col in cols])
                select_snippet = f"{select_snippet1}, {select_snippet2}"
                ordering_snippet = ", ".join([f'"{x}"' for x in columns_present + cols])

                #indexcreation= ""
                #for fk in fks:
                #    indexcreation += f'CREATE INDEX IF NOT EXISTS functable_ind_{name}_{fk} ON "{name}" ({fk});'

                query += f'''DO $$
                -- check if any of the cols is already in there
                BEGIN
                
                                    -- Create the temporary table
                                    CREATE TABLE temp_table as (
                                    SELECT 
                                       {select_snippet}
                                    FROM "{name}" LEFT JOIN "{foreign_table_name}" USING ({", ".join(fks)})
                                    );
                                    
                                    --  Delete current content of the table
                                    DELETE FROM "{name}";
                                
                                    -- Insert improved data into the table
                                    INSERT INTO "{name}"({ordering_snippet})
                                    TABLE temp_table;
                                    
                                    -- Delete the temporary table
                                    DROP TABLE temp_table;

                                END $$;
                                '''

                self.cur.execute(query)
                self.conn.commit()
                self.logger.debug(f"Improved {name}")


        # for all tables, set up foreign keys
        for name in function_tables_settings.keys():
            name = f"{table_prefix}{name}"
            if multicall:
                referenced_table = "multicall"
            else:
                referenced_table = "transactions"
            query = f'ALTER TABLE "{name}" ADD FOREIGN KEY ({", ".join(fks)}) REFERENCES {referenced_table} ON DELETE CASCADE;'
            self.cur.execute(query)

        self.disconnect_databases()
        self.logger.debug(f"function tables improved")

    def remove_codes(self):
        """
        Remove codes table to free up space
        :return:
        """
        self.connect_databases()
        self.logger.debug("Removing codes table")
        query = """
        DROP TABLE IF EXISTS codes;
        """
        self.cur.execute(query)
        self.conn.commit()
        self.disconnect_databases()
        self.logger.debug(f"Codes table removed")

    def add_price_data(self):

        self.logger.debug("Adding price data")
        symbols = ["ETH", "MATIC", "SHIB", "BNB", "WBTC"]
        tokens_substring = ", ".join([f"{x.lower()}_usd float" for x in symbols])
        query = f'''
        DROP TABLE IF EXISTS "priceData";
        CREATE TABLE "priceData" (
            block_id integer PRIMARY KEY,
            {tokens_substring}       
            );
        '''
        get_blockids_query = """
        SELECT block_id, timestamp FROM blocks ORDER BY block_id ASC;
        """

        self.connect_databases()
        self.cur.execute(query)
        self.conn.commit()
        self.cur.execute(get_blockids_query)
        block_df = pd.DataFrame(self.cur.fetchall(), columns=["block_id", "timestamp"])

        col = "timestamp"
        for symbol in symbols:
            newcol = f"{symbol.lower()}_usd"
            path = f"{self.prefix}/data/{symbol}_price.csv"
            price_col = "Close"
            def get_prices(path):
                df_ETH_prices = pd.read_csv(path, index_col=0)
                df_ETH_prices.index.name = "timestamp"
                df_ETH_prices.reset_index(inplace=True)
                df_ETH_prices = df_ETH_prices[["timestamp", price_col]]
                return df_ETH_prices

            df_i = get_prices(path)
            # timestamp 0 means Close price of 0, at it as a new row on top
            df_i = df_i.append({col: 0, price_col: 0}, ignore_index=True)
            df_i = df_i.sort_values(by=col).reset_index(drop=True)

            # timestamp helper array use the timestamp col and round down to the nearest multiple of 3600

            timestamp_helper = pd.DataFrame(block_df[col].unique(), columns=[col])
            timestamp_helper[col] = timestamp_helper[col] - timestamp_helper[col] % 3600

            # lookup price in df_ETH_prices
            timestamp_helper = timestamp_helper.merge(df_i, on=col, how="left")
            block_df[newcol] = timestamp_helper[price_col].values


        block_df.drop(columns=["timestamp"], inplace=True)

        block_df.to_sql("priceData", self.postgres_engine, method=psql_insert_df,if_exists="append", index=False)

        self.disconnect_databases()
        self.logger.debug("Added price data")

    def run(self):

        event_tables_settings = load_table_settings(self.prefix, "event_signatures_with_index.csv")
        function_tables_settings = load_table_settings(self.prefix, "function_signatures_with_index.csv")

        self.postgres_add_blocks()
        self.postgres_add_transactions()
        self.postgres_add_logs()
        self.postgres_add_codes()
        self.add_indices()
        self.create_event_tables(event_tables_settings)
        self.create_function_tables(function_tables_settings)
        self.improve_function_tables(function_tables_settings)
        self.create_multicall_tables(function_tables_settings)
        self.improve_function_tables(function_tables_settings, multicall=True)
        self.create_accounts(event_tables_settings, function_tables_settings)
        self.add_foreign_keys(event_tables_settings, function_tables_settings)
        self.add_price_data()
        self.remove_codes()


if __name__ == "__main__":

    prefix = "../.."
    configs = load_configs(prefix)
    p = Preprocess(configs, prefix)
    #p.add_price_data()
    p.run()
