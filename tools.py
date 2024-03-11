import json
from pathlib import Path
import os
import pandas as pd
import shutil
from mev_inspect.schemas.blocks import Block
from web3 import Web3
from web3 import HTTPProvider
import matplotlib.pyplot as plt
import seaborn as sns
from Crypto.Hash import keccak
import logging
import csv
from io import StringIO
import functools
from copy import deepcopy
import time
import pickle

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

def get_web3(prefix):
    configs = load_configs(prefix=prefix)
    json_path = prefix + "/" + configs["Download"]["PROVIDERS_PATH"]
    provider = "".join(list(load_json(json_path).values())[0])#just pick the first
    web3 = Web3(HTTPProvider(provider))
    return web3

def set_configs(name, prefix):
    # copy settings file from configs to base directory
    shutil.copyfile(f"{prefix}/configs/{name}.toml", f"{prefix}/config_in_use.toml")
    # wait for one second could fix weird bug
    time.sleep(1)

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

def load_mapping(filename,to_col=1, from_col=0):
    df = pd.read_csv(filename)

    # create dictionary from first 2 columns
    mapping = dict(zip(df.iloc[:, from_col], df.iloc[:, to_col]))
    return mapping

def save_json(data, filename):
    dir_ = os.path.dirname(filename)
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_configs(prefix="."):
    return tomllib.loads(Path(prefix + "/config_in_use.toml").read_text(encoding="utf-8"))

def load_configs_by_name(name, prefix="."):
    return tomllib.loads(Path(prefix + "/configs/" + name + ".toml").read_text(encoding="utf-8"))

def flatten(l):
    return [item for sublist in l for item in sublist]


def standard_latex_formatting(tex):

    replacements = {
        r"\$\textbackslash le\$": "$\le$",
        r"\$\textbackslash infty\$": "$\infty$",
        r"\textbackslash \&": "\&",
        "-0.0": "0.0",
    }


    for k, v in replacements.items():
        tex = tex.replace(k, v)

    # thead[Dimension. latexnewline Reduction]-> \thead{Dimension\\Reduction}
    tex = tex.replace(r"thead[[[", r"\thead{")
    tex = tex.replace(r"latexnewline ", r"\\")
    tex = tex.replace(r"]]]", r"}")


    return tex

def latex_save_table(df, filepath, header=True, index=True, column_format=None):
    # round the floats
    df = df.round(3)

    tex = df.to_latex(index=index, header=header, column_format=column_format)
    tex = standard_latex_formatting(tex)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(tex)


def save_prediction_results(df, filename, prefix):
    configs = load_configs(prefix)
    run_name = configs["General"]["run_name"]
    base = f"{prefix}/outputs/{run_name}/predictions"
    filepath = f"{base}/{filename}.csv"
    if not os.path.exists(base):
        os.makedirs(base)
    df.to_csv(filepath)

def save_data(df, filename, prefix):
    configs = load_configs(prefix)
    run_name = configs["General"]["run_name"]
    base = f"{prefix}/outputs/{run_name}/data"
    filepath = f"{base}/{filename}.pkl"
    if not os.path.exists(base):
        os.makedirs(base)
    df.to_pickle(filepath)

def load_data(filename, prefix):
    configs = load_configs(prefix)
    run_name = configs["General"]["run_name"]
    base = f"{prefix}/outputs/{run_name}/data"
    filepath = f"{base}/{filename}.pkl"
    return pd.read_pickle(filepath)

def load_prediction_results(filename, prefix):
    configs = load_configs(prefix)
    run_name = configs["General"]["run_name"]
    base = f"{prefix}/outputs/{run_name}/predictions"
    filepath = f"{base}/{filename}.csv"
    return pd.read_csv(filepath)

def save_table(df, filename, section, prefix, header=True, index=True, column_format=None):
    configs = load_configs(prefix)
    run_name = configs["General"]["run_name"]


    base_csv = f"{prefix}/outputs/{run_name}/tables_raw/{section}"
    base_pkl = f"{prefix}/outputs/{run_name}/tables_pickled/{section}"
    if not os.path.exists(base_csv):
        os.makedirs(base_csv)
    if not os.path.exists(base_pkl):
        os.makedirs(base_pkl)

    df.to_csv(f"{base_csv}/{filename}.csv", index=index)
    df.to_pickle(f"{base_pkl}/{filename}.pkl")

    base = f"{prefix}/outputs/{run_name}/latex_snippets/{section}"
    filepath = f"{base}/{filename}.tex"
    if not os.path.exists(base):
        os.makedirs(base)

    latex_save_table(df, filepath, header=header, index=index, column_format=column_format)


def remake_latex_tables(prefix):
    """TODO This is untested, also deprecated, dont use this, use the Table class instead
    This function allows to remake all the latex tables without having to rerun the whole pipeline
    """

    configs = load_configs(prefix)
    run_name = configs["General"]["run_name"]
    base_pkl = f"{prefix}/outputs/{run_name}/tables_pickled"
    base_latex = f"{prefix}/outputs/{run_name}/latex_snippets"

    for folder in os.listdir(base_pkl):
        for filename in os.listdir(f"{base_pkl}/{folder}"):
            df = pd.read_pickle(f"{base_pkl}/{folder}/{filename}")

            latex_save_table(df, f"{base_latex}/{folder}/{filename[:-4]}.tex")

def save_text(text, filename, chapter, prefix):
    text = str(text)
    configs = load_configs(prefix)
    run_name = configs["General"]["run_name"]
    base = f"{prefix}/outputs/{run_name}/latex_snippets/{chapter}"
    if not os.path.exists(base):
        os.makedirs(base)

    with open(f"{base}/{filename}.tex", "w") as f:
        f.write(text)

def save_data_for_figure(data, filename, chapter, prefix):
    configs = load_configs(prefix)
    run_name = configs["General"]["run_name"]
    base = f"{prefix}/outputs/{run_name}/figtables_pickled/{chapter}"

    if not os.path.exists(base):
        os.makedirs(base)

    with open(f"{base}/{filename}.pkl", "wb") as f:
        pickle.dump(data, f)

def save_figure(filename, chapter, axes, title, x_label, y_label, prefix="."):

    configs = load_configs(prefix)
    run_name = configs["General"]["run_name"]
    base = f"{prefix}/outputs/{run_name}/figures/{chapter}"
    if not os.path.exists(base):
        os.makedirs(base)
    plot_settings = configs["Plotting"]
    title_size = plot_settings["title_size"]
    label_size = plot_settings["label_size"]
    tick_size = plot_settings["tick_size"]

    axes.set_title(title, fontsize=title_size)
    axes.set_xlabel(x_label, fontsize=label_size)
    axes.set_ylabel(y_label, fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(fontsize=label_size)
    plt.savefig(f"{base}/{filename}.svg", dpi=600, bbox_inches='tight')


def load_plotting_settings_config(configs, factor=1):

    sns.set_style('whitegrid')
    plt.rcParams.update({
        "text.usetex": True,
        'font.family': 'serif',
        'font.serif': 'Computer Modern Roman',
        "xtick.bottom": True,
        "ytick.left": True,
        'axes.labelsize': factor* configs["Plotting"]["label_size"],
        'font.size': factor*configs["Plotting"]["font_size"],
        'legend.fontsize': factor*12,
        'xtick.labelsize':factor* configs["Plotting"]["tick_size"],
        'ytick.labelsize': factor*configs["Plotting"]["tick_size"],
        'figure.titlesize':factor* configs["Plotting"]["title_size"],
        'axes.titlesize': factor* configs["Plotting"]["title_size"],
        'figure.dpi': 96,
    })

def load_plotting_settings_static():

    sns.set_style('whitegrid')
    plt.rcParams.update({
        "text.usetex": True,
        'font.family': 'serif',
        'font.serif': 'Computer Modern Roman',
        "xtick.bottom": True,
        "ytick.left": True,
        'axes.labelsize': 12,
        'font.size': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 96,
    })

def load_block(block_path: str) -> Block:
    defaults = {"block_timestamp": 0}

    with open(block_path, "r") as block_file:
        block_json = json.load(block_file)
        return Block(
            **{
                **defaults,
                **block_json,
            }
        )

def get_filehandler(prefix, filename):
    configs = load_configs(prefix)
    fh = logging.FileHandler(f"{prefix}/logs/{filename}.log")
    fh.setLevel(logging.DEBUG)
    run_name = configs["General"]["run_name"]
    formatter = logging.Formatter(f'%(asctime)s - %(name)s - {run_name} - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    return fh


def get_signature_hash(signature:str, trunc=True) -> str:
    # strip the spaces
    # signature = "v2SwapExactInput(address,uint256,uint256,address[],address)"
    signature = signature.replace(" ", "")
    k = keccak.new(digest_bits=256)
    sig = bytearray()
    sig.extend(map(ord, signature))
    k.update(sig)
    if trunc:
        return "0x" + k.hexdigest()[:8]
    else:
        return "0x" + k.hexdigest()



def psql_insert_df(table, conn, keys, data_iter):
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY "{}" ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)



def log_time(function):
    """
    Decorator for time measurement and logging
    :param function:
    :return:
    """

    def wrapper(self, *args, **kwargs):
        start = time.time()
        self.logger.debug(f"Starting {function.__name__}")
        result = function(self, *args, **kwargs)
        end = time.time()
        self.logger.debug(f"Finished {function.__name__} in {end - start} seconds")
        return result

    return wrapper


def lru_cache(maxsize=10000, typed=False, copy=True):
    if not copy:
        return functools.lru_cache(maxsize, typed)
    def decorator(f):
        cached_func = functools.lru_cache(maxsize, typed)(f)
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return deepcopy(cached_func(*args, **kwargs))
        return wrapper
    return decorator


def feature_name_mapper_single(featurename):
    """
    Maps internal feature name to something more presentable

    replace __ with -
    turn snake case to camel case

    replace outtime with out
    replace intime with in

    :param featurename:
    :return:
    """
    #featurename = "tx__time__outtime_hourly_entropy"

    featurename = featurename.replace("outtime", "out")
    featurename = featurename.replace("intime", "in")
    featurename = featurename.replace("transaction_index_relative", "tx_index_rel")
    featurename = featurename.replace("custom__n_tx_per_block", "block__n_tx_per_block")
    featurename = featurename.replace("generic__", "")

    generic_names = ["mean", "median", "std", "min", "max","quantile_95"]


    for generic_name in generic_names:
        featurename = featurename.replace(f"__{generic_name}", f"_{generic_name}")

    featurename = featurename.replace("__", "-")
    featurename = featurename.replace("_", " ")
    featurename = featurename.title()
    featurename = featurename.replace(" ", "")
    return featurename

def feature_name_mapper(featurenames):

    return [feature_name_mapper_single(featurename) for featurename in featurenames]


def get_window_names_mapping(configs):

    window_names_internal = configs["General"]["window_names_internal"]
    window_names = configs["General"]["window_names"]
    window_names_mapping = {window_names_internal[x]: window_names[x] for x in range(len(window_names))}
    return window_names_mapping