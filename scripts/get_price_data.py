import ccxt
import pandas as pd

def get_all_price_data(prefix):
    get_ETH_price(prefix)
    get_SHIB_price(prefix)
    get_BNB_price(prefix)
    get_MATIC_price(prefix)
    get_WBTC_price(prefix)

def get_ETH_price(prefix):
    symbol = 'ETH/USDT'  # initial symbol
    name = "ETH"
    get_price_data(prefix, symbol, name)

def get_SHIB_price(prefix):
    symbol = 'SHIB/USDT'  # initial symbol
    name = "SHIB"
    get_price_data(prefix, symbol, name)

def get_BNB_price(prefix):
    symbol = 'BNB/USDT'  # initial symbol
    name = "BNB"
    get_price_data(prefix, symbol, name)

def get_MATIC_price(prefix):
    symbol = 'MATIC/USDT'  # initial symbol
    name = "MATIC"
    get_price_data(prefix, symbol, name)

def get_WBTC_price(prefix):
    symbol = 'BTC/USDT'  # initial symbol
    name = "WBTC"
    get_price_data(prefix, symbol, name)


def get_price_data(prefix, symbol, name):
    exch = 'binance'  # initial exchange
    t_frame = '1h'  # 1-day timeframe, usually from 1-minute to 1-week depending on the exchange

    exchange = getattr(ccxt, exch)()
    exchange.load_markets()
    data = exchange.fetch_ohlcv(symbol, t_frame, since=0)
    header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame(data, columns=header).set_index('Timestamp')
    max_date_number = df.index.max()
    # represent unix as date
    max_date = pd.to_datetime(max_date_number, unit='ms')
    dfs = [df]
    i = 0
    while True:
        print(f"Fetching data for {symbol} from {max_date}")

        data = exchange.fetch_ohlcv(symbol, t_frame, since=max_date_number)
        df_new = pd.DataFrame(data, columns=header).set_index('Timestamp').iloc[1:] # first one is always a duplicate
        dfs.append(df_new)
        max_date_number = df_new.index.max()
        max_date = pd.to_datetime(max_date_number, unit='ms')

        if len(df_new) < 499:
            break

    df_all = pd.concat(dfs)
    df_all.index = df_all.index / 1000  # Timestamp is 1000 times bigger than it should be in this case
    df_all.index = df_all.index.astype(int)
    timestamps_test = pd.Series(df_all.index)
    diffs = timestamps_test.diff()
    min_, max_ = timestamps_test.min(), timestamps_test.max()
    # get the numbers inbetween in 3600 steps
    timestamps_clean = pd.Series(range(min_, max_, 3600))
    # insert rows with NaNs
    df_all = df_all.reindex(timestamps_clean, fill_value=None)
    # average with the previous and next value
    #mask = df_all.isnull().any(axis=1)
    df_all = df_all.interpolate(method='linear')
    df_all['Date'] = pd.to_datetime(df_all.index, unit='s')

    filename = f"{prefix}/data/{name}_price.csv"
    df_all.to_csv(filename)

if __name__ == '__main__':
    prefix = ".."
    get_all_price_data(prefix)
