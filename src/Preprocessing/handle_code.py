import pandas as pd
from tqdm import tqdm
import os
import time
import sqlite3
import csv
import logging
import numpy as np

logger = logging.getLogger('preprocessing.handle_code')

def filter_addresses(prefix_db):
    # the trace files are pretty big and contain the addresses we will consider in this analysis
    # from_address and to_address are the columns that we are interested in
    traces_folder = f"{prefix_db}/erigon_extract/uncompressed/traces"
    outfile = f"{prefix_db}/preprocessed/addresses.csv"

    traces_files = os.listdir(traces_folder)
    traces_files = [f"{traces_folder}/{file}" for file in traces_files]

    logger.debug(f"Found {len(traces_files)} trace files")
    start = time.time()
    # continuously load the addresses from the traces files and add them to the set
    addresses = set()
    for file in tqdm(traces_files):
        df = pd.read_csv(file, usecols=["from_address", "to_address"])
        addresses.update(df["from_address"].unique())
        addresses.update(df["to_address"].unique())

    logger.debug(f"Found {len(addresses)} addresses. Took {time.time() - start}")

    pd.DataFrame({"address": list(addresses)}).to_csv(outfile, index=False)

    return outfile


def join_address_code(prefix_db):
    # now join addresses with the code and create the nodes
    # the code file is huge, so first index it and then join it with the addresses
    outfile = f"{prefix_db}/preprocessed/accounts/addresses_with_code.csv"
    address_path = f"{prefix_db}/preprocessed/addresses.csv"

    codes_path = f"{prefix_db}/erigon_extract/uncompressed/codes/trace_creations.csv"

    relevant_columns = ["to_address", "output"]

    db_path = f"{prefix_db}/db.sqlite3"
    # Connect to the database (or create a new file if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    start = time.time()

    cursor.execute("DROP TABLE IF EXISTS Codes")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Codes (
        to_address TEXT,
        output TEXT
    )
    """)

    # read into pandas in chunks and save to sqlite
    chunksize = 5 * 10 ** 6
    for chunk in tqdm(pd.read_csv(codes_path, usecols=relevant_columns, chunksize=chunksize)):
        chunk.to_sql("Codes", conn, if_exists="append", index=False)

    logger.debug(f"Created and filled Code table in {time.time() - start}")
    start = time.time()
    # create index
    cursor.execute("CREATE INDEX idx_to_address ON Codes (to_address)")
    logger.debug(f"Created index in {time.time() - start}")

    start = time.time()
    # drop table if it exists
    cursor.execute("DROP TABLE IF EXISTS Addresses")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Addresses (
      address TEXT
    )
    """)

    # read into pandas in chunks and save to sqlite
    chunksize = 10 ** 6
    for chunk in tqdm(pd.read_csv(address_path, chunksize=chunksize)):
        chunk.to_sql("Addresses", conn, if_exists="append", index=False)

    logger.debug(f"Created and filled Addresses table in {time.time() - start}")
    start = time.time()

    # create index
    cursor.execute("CREATE INDEX idx_address ON Addresses (address)")
    logger.debug(f"Created index in {time.time() - start}")


    # Commit the changes and close the connection
    conn.commit()
    conn.close()



    # Reopen the connection to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    start = time.time()

    query = """
    SELECT *
    FROM Addresses
    LEFT JOIN Codes
    ON Addresses.address = Codes.to_address
    """

    cursor.execute(query)


    logger.debug(f"Left join executed in {time.time() - start}")
    start = time.time()
    logger.debug("Writing to file...")
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([i[0] for i in cursor.description])
        writer.writerows(cursor)


    logger.debug(f"Time elapsed writing: {time.time() - start}")


    # Close the connection
    conn.close()

    # delete database
    os.remove(db_path)


def clean_addresses(prefix_db):
    outfile = f"{prefix_db}/preprocessed/accounts/addresses_with_code.csv"
    df = pd.read_csv(outfile, usecols=["address", "output"])
    df = df.dropna(subset=["address"])
    df = df.drop_duplicates(subset=["address"], keep="last")  # use most up to date code, assumes that left join keeps the order
    df["type"] = df["output"].apply(lambda x: "EOA" if x is np.nan else "CA")
    df.to_csv(outfile, index=False)
    logger.debug(f"Cleaned addresses_with_code file")
