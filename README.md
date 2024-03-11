# Ethereum Bot Detection
This code is used was used to generate artifacts of the paper 
"Detecting Financial Bots on the Ethereum Blockchain".

Note that the methods used in this paper are compute intensive for a single PC and require 64GB of RAM.
On an AMD Ryzen 5 2600 Six-Core Processor (3.4 GHz) with 64GB of RAM, the code takes about 24 hours to run.

### How to install
This repository has only been tested on Windows 10 but may also work Linux distributions.

- install miktex and add the ...\MiKTeX\miktex\bin\x64\ folder to the path
- install postgres 15.1 and adjust the pg_hba.conf file as laid out here:  https://stackoverflow.com/questions/64210167/unable-to-connect-to-postgres-db-due-to-the-authentication-type-10-is-not-suppor


We use MEV-inspect https://github.com/flashbots/mev-inspect-py/tree/main/mev_inspect with slightly adapted code
to work with data provided by graphsense-lib https://github.com/graphsense/graphsense-lib.

# todo for faster checking of adequate data, we provide a sample of the data under XXXXTODOXXXX

# populate requirements.txt
