
<h1 align="center">Ethereum</h1>
<p align="center"> Bot Detection
  This code was used to generate results presented in the paper <a href="https://arxiv.org/abs/2403.19530">"Detecting Financial Bots on the Ethereum Blockchain"</a>.
</p>
<p align="center">
  <img src="assets/bot.webp" alt="Bot Image width="400" height="400"">
</p>

Note that the methods used in this paper are compute intensive for a single PC and require 64GB of RAM.
On an AMD Ryzen 5 2600 Six-Core Processor (3.4 GHz) with 64GB of RAM, the code takes about 24 hours to run and in
addition to the results produced for the paper, several other tables and figures are generated.

### How to install
This repository has only been tested on Windows 10 but may also work on Linux distributions.

- install miktex and add the `...\MiKTeX\miktex\bin\x64\` folder to the path
- install `postgres 15.1` and adjust the `pg_hba.conf` file as laid out here:  https://stackoverflow.com/questions/64210167/unable-to-connect-to-postgres-db-due-to-the-authentication-type-10-is-not-suppor

To install the required python packages use the following commands

```
python -m venv .venv
.venv/Scripts/activate source 
pip install requirements.txt
```
or on Linux (untested)
```
python -m venv .venv
source .venv/bin/activate
pip install requirements.txt
```


### Minimal Example

##### Setup
To use this repository block, transaction, and log data (enriched as provided by graphsense-lib) are required

To use it, point `PREFIX_DB` in `configs/test.toml` to the folder containing the raw data as demonstrated in 
a test sample provided in `test_data/test_run`.

Furthermore, a trace_creations.csv file is required that contains all addresses that should be marked as a smart contract
A minimal example is also provided in `test_data/codes`. Note that the content of the output column is not of central
importance. As soon as an address appears in the to_address column, it will be marked as a smart contract.

The data is provided to the program in a compressed format as demonstrated in the minimal example.

##### Run

To run the code use the following command
```
python pipeline.py
```

Tables and figures will be saved in the `output` folder.


### Acknowledgements

We use MEV-inspect https://github.com/flashbots/mev-inspect-py/tree/main/mev_inspect with slightly adapted code
to work with data provided by graphsense-lib https://github.com/graphsense/graphsense-lib.
