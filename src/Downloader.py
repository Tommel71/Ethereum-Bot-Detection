"""The purpose of this script is that I can query raw transaction data from the blockchain and store it in a
  MongoDB database. Specify a new blockrange to download what is not already in the database.

  Also: Scrape the 4byte database for signatures and store them as a json file in the data folder.
  """

from tools import load_json, save_text
import time
import pymongo
import pandas as pd
from tools import load_configs
import requests
from tqdm import tqdm

class DownloadPipeline:

    def __init__(self, configs, prefix=""):
        self.configs = configs
        self.DB_NAME = configs["Download"]["DB_NAME"] + "_" + configs["General"]["run_name"]
        self.PROVIDERS_PATH = configs["Download"]["PROVIDERS_PATH"]
        self.prefix = prefix

        blockrange = configs["Download"]["blockrange"]
        self.block_numbers_to_get = list(range(blockrange[0], blockrange[1]))
        credential_list = list(load_json(f"{self.prefix}/{self.PROVIDERS_PATH}").values())


    def int_to_string(self, mapping, names):
        for name in names:
            # have to cast long values to string in mongodb
            mapping[name] = str(mapping[name])
        return mapping

    def hexbytes_to_hex(self, element):

        # convert attributedict to dict
        if element.__class__.__name__ == "AttributeDict":
            element = dict(element)

        if isinstance(element, dict):
            for k, v in element.items():
                element[k] = self.hexbytes_to_hex(v)

        elif isinstance(element, list):
            for i, v in enumerate(element):
                element[i] = self.hexbytes_to_hex(v)

        elif isinstance(element, bytes):
            element = str(element.hex())

        return element


    def whatsthere_whatsmissing(self, db_collection, to_get):
        data_whats_there = list(db_collection.find({"_id": {"$in": to_get}}))
        whats_there = [b["_id"] for b in data_whats_there]
        whats_missing = list(set(to_get) - set(whats_there))
        return data_whats_there, whats_missing

    def scrape_4byte(self, events=True):
        """scrape 4byte for signatures, takes about 2.5h for the ~800k signatures
            has to be run in one go
        """
        if events:
            collection = "event_signatures"
            name = "event-signatures"
        else:
            collection = "signatures"
            name = "signatures"

        self.db[collection].drop()

        next = f"https://www.4byte.directory/api/v1/{name}/?format=json&ordering=created_at&page=1"
        page = 1
        while next is not None:
            try:
                print(f"page {page}")
                r = requests.get(next)
                data = r.json()
                results = data["results"]
                hex_and_text = [{"_id":results[i]["id"], "hex_signature": results[i]["hex_signature"], "text_signature" :results[i]["text_signature"]} for i in range(len(data["results"]))]

                self.db[collection].insert_many(hex_and_text)
                next = data["next"]
                page+=1


            except Exception as exc:
                print(exc)
                print(r)
                try:
                    print(r.json())
                except:
                    pass

                print("sleeping for 10s")
                time.sleep(20)

    def signature_to_text(self):
        """Load signatures from the db and get the first entry for each hex signatures in the database. We query in order so thats the first ever added to 4byte"""

        def create_signature_mapping(collection):
            """create a mapping from hex signature to text signature and save it as json"""
            signatures = self.db[collection].find()
            df = pd.DataFrame(signatures)
            n_duplicate_hex = df["hex_signature"].duplicated().sum()
            df.groupby("hex_signature").first().drop(columns=["_id"]).to_csv(f"{self.prefix}/data/{collection}.csv")

            save_text(str(n_duplicate_hex), f"n_duplicates_signature_hex_{collection}", "data", prefix=self.prefix)

        create_signature_mapping("event_signatures")
        create_signature_mapping("signatures")

    def connect_databases(self):

        self.client = pymongo.MongoClient()
        self.db = self.client[self.DB_NAME]


    def disconnect_databases(self):
        self.client.close()

    def execute(self):
        self.connect_databases()

        # split up self.block_numbers_to_get into chunks of 5
        block_numbers_to_get = [self.block_numbers_to_get[i:i + 5] for i in range(0, len(self.block_numbers_to_get), 5)]
        for blocks_to_get in tqdm(block_numbers_to_get):
            print(f"getting blocks {blocks_to_get}")
            ### QUERY BLOCKS ###
            transaction_hashes_to_get = self.get_blocks(blocks_to_get)

            ### QUERY TRANSACTIONS ###
            addresses_to_get = self.get_transactions(transaction_hashes_to_get)

            ### QUERY RECEIPTS ###
            self.get_transaction_receipts(transaction_hashes_to_get)

            ### QUERY CODE ###
            self.get_codes(addresses_to_get)

        if self.configs["Download"]["scrape_4byte"]:
            self.scrape_4byte(events=False)
            self.scrape_4byte(events=True)
            self.signature_to_text()

        self.disconnect_databases()

    def evaluate_download(self):
        """
        Get statistics about the data downloaded from the blockchain and save it as a csv and as a latex table
        """
        print("asd")
        #self.db.blocks.aggregate()

if __name__ == "__main__":
    download_configs = load_configs("..")

    pipeline = DownloadPipeline(download_configs, prefix="..")
    pipeline.evaluate_download()
    #pipeline.execute()