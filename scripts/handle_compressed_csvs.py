"""
Script to unpack the CSV files from the gz files from the erigon extract.
"""

import logging
from pathlib import Path
from tqdm import tqdm
import shutil
import gzip
from tools import load_configs


def unpack_from_gz(gz_path: str, out_dir: str) -> None:
    """
    Unpack the CSV files from the gz files from the erigon extract.
    """
    print("Unpacking CSV files from gz files", gz_path)

    gz_path = Path(gz_path)
    out_dir = Path(out_dir)

    # Get the list of gz files
    gz_files = list(gz_path.glob("*.gz"))
    logging.info(f"Found {len(gz_files)} gz files")

    # Create the output directory
    out_dir.mkdir(exist_ok=True, parents=True)

    # Unpack the CSV files
    for gz_file in tqdm(gz_files):
        unpack_gz(gz_file, out_dir)

    print(gz_path, "done")


def unpack_gz(gz_file: Path, out_dir: Path) -> None:
    """
    Unpack the CSV files from a single gz file.
    """
    # Get the name of the gz file
    gz_name = gz_file.stem

    # Create the output directory if it doesnt exist yet

    out_dir_gz = out_dir / gz_name
    if not out_dir.exists():
        out_dir.mkdir()


    # Unpack the CSV files
    with gzip.open(gz_file, "rb") as f_in:
        with open(out_dir_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

def unpack_all(prefix):
    configs = load_configs(prefix)
    prefix_db = configs["General"]["PREFIX_DB"]
    base = f"{prefix_db}/erigon_extract"
    types = ["blocks", "transactions", "logs"] # , "codes"]  # + ["traces"]
    in_dirs = [base + "/compressed/" + type_ for type_ in types]
    out_dirs = [base + "/uncompressed/" + type_ for type_ in types]

    for in_dir, out_dir in zip(in_dirs, out_dirs):
        unpack_from_gz(in_dir, out_dir)

    # codes is special because they are large and I want them to share a folder and not be repeated in all the dirs
    # also i need the same file for all the dirs, different to logs, blocks transactions
    in_dir = f"{prefix_db}/../codes_for_large/"
    out_dir = f"{base}/uncompressed/codes"
    unpack_from_gz(in_dir, out_dir)

def delete_all_unpacked(prefix):
    configs = load_configs(prefix)
    prefix_db = configs["General"]["PREFIX_DB"]
    base = f"{prefix_db}/erigon_extract"
    types = ["blocks", "transactions", "logs", "codes"]  # + ["traces"]
    out_dirs = [base + "/uncompressed/" + type_ for type_ in types]
    for out_dir in out_dirs:
        shutil.rmtree(out_dir, ignore_errors=True)

if __name__ == "__main__":
    prefix = ".."
    unpack_all(prefix)