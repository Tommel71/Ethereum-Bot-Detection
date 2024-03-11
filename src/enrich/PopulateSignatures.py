"""Populates MongoDB with scraped data from 4byte"""
import pymongo
from tools import load_configs

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

class PopulateSignatures:

    def __init__(self, configs, prefix=""):


        self.preprocessing_configs = configs["Preprocessing"]
        self.DB_NAME = configs["Download"]["DB_NAME"]
        self.PROVIDERS_PATH = configs["Download"]["PROVIDERS_PATH"]
        self.prefix = prefix

    def dosth(self):
        pass

    def connect_databases(self):

        self.client = pymongo.MongoClient()
        self.db = self.client[self.DB_NAME]


    def disconnect_databases(self):
        self.client.close()


if __name__ == "__main__":
    configs = load_configs("../..")
    p = PopulateSignatures(configs, "../..")


topics = ["0x309bb360e8b69c23937ccc5fb01f9aeeead1c95a99604e175113ff82f2b1723a","0x000000000000000000000000e1d9b2b918e7bf4c27b108c7d78c839356b7c69f","0x00000000000000000000000035e4876102389f13d78381d317ff4612412a56c9","0x0000000000000000000000000000000000000000000000000000000000000360"]
address ="0xF0542Ed44d268C85875b3B84B0e7Ce0773E9aEEf"
# 0xc2f74145a896b6a5b938216d130f5c382d3d77d5d501ab41e43681cbebcf3b97  tx hash
signature_hash = "0x309bb360e8b69c23937ccc5fb01f9aeeead1c95a99604e175113ff82f2b1723a"
signature_text = "Registration(address,address,uint256,uint256)"
data = "0x000000000000000000000000000000000000000000000000000000000000005d"
mapping = {signature_hash:signature_text}
