
<h1 align="center">Ethereum Bot Detection</h1>
<p align="center">
  This code was used to generate results presented in the paper <a href="https://arxiv.org/abs/2403.19530">"Detecting Financial Bots on the Ethereum Blockchain"</a>.
</p>
<p align="center">
  <img src="assets/bot.webp" alt="Bot Image width="400" height="400"">
</p>

Note that the methods used in this paper are compute intensive for a single PC and require 64GB of RAM.
On an AMD Ryzen 5 2600 Six-Core Processor (3.4 GHz) with 64GB of RAM, the code takes about 24 hours to run and in
addition to the results produced for the paper, several other tables and figures are generated.



# Minimal Example

We provide two types to run the pipeline with a reduced dataset. To run it with more data acquire more using `graphsense-lib`
and adjust the `PREFIX_DB` in `configs/test.toml` to the folder containing the raw data.

## Docker

Use docker compose to start the pipeline on the provided test data.

`docker-compose up`

Running this may take some time. In the end, the results will be saved in the `output` folder.


## Local

### How to install
This repository has only been tested on Windows 10 but may also work on Linux distributions.

- install miktex and add the `...\MiKTeX\miktex\bin\x64\` folder to the path
- install `postgres 15.1` and adjust the `pg_hba.conf` file as laid out here:  https://stackoverflow.com/questions/64210167/unable-to-connect-to-postgres-db-due-to-the-authentication-type-10-is-not-suppor

We use `uv` for dependency management. Run the following commands to install the dependencies

```
uv python install 3.10
uv sync
```


##### Setup
To use this repository block, transaction, and log data (enriched as provided by graphsense-lib) are required

To use it, point `PREFIX_DB` in `configs/test.toml` to the folder containing the raw data as demonstrated in 
a test sample provided in `test_data/test_run`. Change the `credentials/postgres/test.json` file `host` variable to localhost.

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
