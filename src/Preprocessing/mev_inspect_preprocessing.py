from src.Preprocessing.create_mev_inspect_predictions import csv_to_erigon_folder, precompute_mev_inspect
from tools import load_configs, get_filehandler
import logging

prefix = "../.."

logger = logging.getLogger("preprocessing")
logger.setLevel(logging.DEBUG)
fh = get_filehandler(prefix, "preprocessing")
logger.addHandler(fh)


configs = load_configs(prefix)
prefix_db = configs["General"]["PREFIX_DB"]
csv_to_erigon_folder(f"{prefix_db}/erigon_extract/uncompressed/traces",
                       f"{prefix_db}/erigon_extract/uncompressed/logs",
                       f"{prefix_db}/preprocessed/mev_inspect_blocks",
                       f"{prefix}/data/mev_inspect_template.json")


precompute_mev_inspect(f"{prefix_db}/preprocessed/mev_inspect_blocks",
                       f"{prefix_db}/preprocessed/mev_inspect_predictions")