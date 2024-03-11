"""
Price data from https://data.nasdaq.com/data/BITFINEX/ETHUSD-ethusd-exchange-rate

"""
import pandas as pd
from tools import save_data_for_figure

def run(prefix):
    df_price = pd.read_csv(f"{prefix}/data/BITFINEX-ETHUSD.csv")
    df_price['Date'] = pd.to_datetime(df_price['Date'])  # Convert 'Date' column to datetime
    df_price.set_index('Date', inplace=True)  # Set 'Date' as the index
    filename, chapter = "timeframes_of_interest", "results"
    save_data_for_figure(df_price, filename, chapter, prefix)


if __name__ == "__main__":
    prefix = ".."
    run(prefix)