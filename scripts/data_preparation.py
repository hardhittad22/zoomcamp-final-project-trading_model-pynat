# imports
import numpy as np
import pandas as pd
import requests
import logging
import pickle

# finance
import yfinance as yf
import pandas_ta as ta

# visualisation
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# time
import time
from datetime import date, datetime, timedelta



def get_data():

    """# 0. Get Data

    ## 0.1 Get Crypto
    """

    def get_coins():
        # Set up logging to display info and error messages
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # List of coins
        coins = [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "DOGEUSDT",
            "ADAUSDT",
            "TRXUSDT",
            "ARBUSDT",
            "SHIBUSDT",
            "WBTCUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "BCHUSDT",
            "UNIUSDT",
            "NEARUSDT",
            "LTCUSDT",
            "FILUSDT",
            "BEAMUSDT",
            "APTUSDT",
            "AVAXUSDT",
            "MATICUSDT",
            "ATOMUSDT",
            "XLMUSDT",
            "RNDRUSDT",
        ]

        def fetch_crypto_ohlc(coin, interval="1h", start_str="1 Jan, 2020"):
            url = "https://api.binance.com/api/v1/klines"

            # Convert start date string to datetime object
            start_time = pd.to_datetime(start_str)
            end_time = datetime.now()

            all_data = []

            while start_time < end_time:
                # Define request parameters for Binance API
                params = {
                    "symbol": coin,
                    "interval": interval,
                    "startTime": int(start_time.timestamp() * 1000),
                    "endTime": int(
                        (start_time + timedelta(days=30)).timestamp() * 1000
                    ),  # Fetching 30 days at a time
                }

                response = requests.get(url, params=params)

                # Check if request was successful
                if response.status_code != 200:
                    logging.error(f"Error fetching data for {coin}: {response.status_code}")
                    break

                data = response.json()

                # Check if data is returned
                if not data:
                    logging.warning(f"No OHLC data found for {coin}.")
                    break

                all_data.extend(data)

                # Update start time for next request
                start_time = pd.to_datetime(data[-1][0], unit="ms") + timedelta(
                    milliseconds=1
                )

                logging.info(f"Fetched data for {coin} up to {start_time}")
                time.sleep(0.5)  # Pause to avoid API limits

            if not all_data:
                return pd.DataFrame()

            # Convert fetched data into a DataFrame with appropriate column names
            ohlc_df = pd.DataFrame(
                all_data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )

            # Convert timestamp to datetime and set it as index
            ohlc_df["Date"] = pd.to_datetime(ohlc_df["timestamp"], unit="ms")
            ohlc_df["Ticker"] = coin
            ohlc_df["Ticker_Type"] = "Cryptocurrency"

            # Convert price and volume columns to float
            ohlc_df[["Open", "High", "Low", "Close", "Volume"]] = ohlc_df[
                ["open", "high", "low", "close", "volume"]
            ].astype(float)
            ohlc_df["Adj Close"] = ohlc_df["close"]

            # Setting the timezone to Berlin
            ohlc_df["Date"] = (
                ohlc_df["Date"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
            )
            ohlc_df.set_index("Date", inplace=True)

            return ohlc_df[
                [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                    "Ticker",
                    "Ticker_Type",
                ]
            ]

        def generate_datetime_features(df):
            # Generate additional datetime features
            df["Year"] = df.index.year.astype(int)
            df["Month"] = df.index.month.astype(int)
            df["Weekday"] = df.index.weekday.astype(int)
            df["Hour"] = df.index.hour.astype(int)
            df["Coin"] = df["Ticker"].str.replace("USDT", "")  # Extracting Name

            return df

        # Main Script
        all_data = pd.DataFrame()

        for coin in coins:
            logging.info(f"Fetching data for {coin}")
            df = fetch_crypto_ohlc(coin)

            if not df.empty:
                df = generate_datetime_features(df)
                all_data = pd.concat([all_data, df])
            else:
                logging.warning(f"No data fetched for {coin}")

        # Reorder columns to have datetime features at the beginning
        cols = ["Year", "Month", "Weekday", "Hour", "Ticker", "Ticker_Type"] + [
            col
            for col in all_data.columns
            if col not in ["Year", "Month", "Weekday", "Hour", "Ticker", "Ticker_Type"]
        ]
        all_data = all_data[cols]

        # Save the final DataFrame to a CSV file
        all_data.to_csv('crypto.csv', index=True)
        logging.info("Data fetching and processing complete. Data saved to crypto.csv")


    if __name__ == "__main__":
        get_coins()


    """## 0.2 Get Stocks"""

    # Function to download historical stock data
    def get_stats(ticker, period="2y", interval="1h"):
        data = yf.download(ticker, period=period, interval=interval)
        return data


    # Adding Date and Time Features
    def add_date_features_and_shifts(df):
        # Convert index to datetime and localize to Berlin timezone
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert("Europe/Berlin")
        df.set_index("Date", inplace=True)
        # Extract year, month, weekday, and hour from the datetime index
        df["Year"] = df.index.year.astype(int)
        df["Month"] = df.index.month.astype(int)
        df["Weekday"] = df.index.weekday.astype(int)
        df["Hour"] = df.index.hour.astype(int)


    def top_stocks():
        # Getting top stocks from yahoofinance
        yfinance_most_active_stocks_web = pd.read_html(
            "https://finance.yahoo.com/most-active/?offset=0&count=100"
        )
        FIELDS = ["Symbol", "Name", "Volume"]
        stock_tickers = yfinance_most_active_stocks_web[0][FIELDS]

        # Limit to top 25 most active stocks
        stock_tickers = stock_tickers.head(25)

        # Initialize an empty DataFrame for storing stock data
        stocks_df = None
        period = "2y"
        interval = "1h"

        # Loop through each stock ticker and download data
        for elem in stock_tickers.Symbol.to_list():
            print(f"Downloading stats for stock: {elem}")
            one_ticker_df = get_stats(elem, period=period, interval=interval)
            if one_ticker_df.empty:
                print(f"No data available for stock: {elem}")
                continue
            # Add additional columns to the DataFrame
            one_ticker_df["Date"] = one_ticker_df.index
            one_ticker_df["Ticker"] = elem
            one_ticker_df["Ticker_Type"] = "Stock"
            # Concatenate each stock's data into a single DataFrame
            if stocks_df is None:
                stocks_df = one_ticker_df
            else:
                stocks_df = pd.concat([stocks_df, one_ticker_df], ignore_index=True, axis=0)
            time.sleep(0.5)  # pause to avoid API overload

        # Adding date and time features
        add_date_features_and_shifts(stocks_df)

        # Reorder columns to have date and time features at the beginning
        cols = ["Year", "Month", "Weekday", "Hour", "Ticker", "Ticker_Type"] + [
            col
            for col in stocks_df.columns
            if col not in ["Year", "Month", "Weekday", "Hour", "Ticker", "Ticker_Type"]
        ]
        stocks_df = stocks_df[cols]

        # save DataFrame as csv
        stocks_df.to_csv('stocks.csv', index=True)

        # Print the first few rows
        print(stocks_df.head())


    if __name__ == "__main__":
        top_stocks()



    """# 1. One DataFrame

    ## 1.1 Calculations on DataFrame
    """


    # Function to add technical indicators to the DataFrame
    def add_ta_indicators(df):
        # Ensure the columns have the correct data types
        df["Open"] = df["Open"].astype("float64")
        df["High"] = df["High"].astype("float64")
        df["Low"] = df["Low"].astype("float64")
        df["Close"] = df["Close"].astype("float64")
        df["Volume"] = df["Volume"].astype("float64")

        # Calculate various technical indicators
        df.ta.adx(high="High", low="Low", close="Close", append=True)
        df.ta.aroon(high="High", low="Low", append=True)
        df.ta.bop(open="Open", high="High", low="Low", close="Close", append=True)
        df.ta.cci(high="High", low="Low", close="Close", append=True)
        df.ta.cmo(close="Close", append=True)
        df.ta.macd(close="Close", append=True)
        df.ta.mom(close="Close", append=True)
        df.ta.ppo(close="Close", append=True)
        df.ta.roc(close="Close", append=True)
        df.ta.rsi(close="Close", append=True)
        df.ta.stoch(high="High", low="Low", close="Close", append=True)
        df.ta.willr(high="High", low="Low", close="Close", append=True)
        df.ta.bbands(close="Close", append=True)
        df.ta.dema(close="Close", append=True)
        df.ta.ema(close="Close", append=True)
        df.ta.midpoint(close="Close", append=True)
        df.ta.midprice(high="High", low="Low", append=True)
        df.ta.sma(close="Close", append=True)
        df.ta.tema(close="Close", append=True)
        df.ta.wma(close="Close", append=True)
        df.ta.atr(high="High", low="Low", close="Close", append=True)
        df.ta.trix(close="Close", append=True)
        df.ta.kama(close="Close", append=True)
        df.ta.psar(high="High", low="Low", close="Close", append=True)

        return df


    # Function to add custom calculations to the DataFrame
    def add_custom_calculations(df, group_by_col="Ticker"):
        new_columns = {}

        # Shifted Close values for previous and next periods
        new_columns["Adj_Close_Minus_1"] = df.groupby(group_by_col)["Close"].shift(-1)
        new_columns["Adj_Close_Plus_1"] = df.groupby(group_by_col)["Close"].shift(1)

        # Growth calculations for different time periods
        for hours in [1, 4, 24, 48, 72, 168, 336, 720]:
            new_columns[f"Growth_{hours}h"] = (
                df.groupby(group_by_col)["Close"].shift(hours) / df["Close"]
            )
            future_shifted = df.groupby(group_by_col)["Close"].shift(-hours)
            new_columns[f"Growth_Future_{hours}h"] = future_shifted / df["Close"]
            new_columns[f"Is_Positive_Growth_{hours}h_Future"] = np.where(
                new_columns[f"Growth_Future_{hours}h"] > 1, 1, 0
            )

        # Fibonacci levels calculation
        new_columns["Fibonacci_0"] = df["Adj Close"]
        new_columns["Fibonacci_23_6"] = df["High"] - (df["High"] - df["Low"]) * 0.236
        new_columns["Fibonacci_38_2"] = df["High"] - (df["High"] - df["Low"]) * 0.382
        new_columns["Fibonacci_50"] = df["High"] - (df["High"] - df["Low"]) * 0.5
        new_columns["Fibonacci_61_8"] = df["High"] - (df["High"] - df["Low"]) * 0.618
        new_columns["Fibonacci_100"] = df["Low"]

        # Assign new columns to the DataFrame
        df = df.assign(**new_columns)

        return df


    # Function to fetch market indices data from Yahoo Finance
    def fetch_market_indices(market_indices):
        indices_data = {}
        for index in market_indices:
            # Download historical data for each index
            data = yf.download(index, period="1y", interval="1h")
            # Convert timezone and round to the nearest 30 minutes
            data.index = data.index.tz_convert("Europe/Berlin").round("30min")
            # Keep only the adjusted close price and rename it to the index symbol
            data = data[["Adj Close"]].rename(columns={"Adj Close": index})
            indices_data[index] = data
        # Combine all indices data into a single DataFrame
        combined_indices = pd.concat(indices_data.values(), axis=1)
        return combined_indices


    def calculations():
        # Load CSV files
        crypto_df = pd.read_csv('crypto.csv')
        stocks_df = pd.read_csv('stocks.csv')

        #making sure nothing got lost
        unique_coins_before = crypto_df["Ticker"].nunique()
        unique_stocks_before = stocks_df["Ticker"].nunique()
        print(f"Number of unique ticker before combining: {unique_coins_before}")
        print(f"Number of unique ticker before combining: {unique_stocks_before}")

        # Concatenate the DataFrames
        combined_df = pd.concat([crypto_df, stocks_df], ignore_index=True)

        # After merging and processing
        num_unique_coins_after = combined_df[
            combined_df["Ticker_Type"] == "Cryptocurrency"
        ]["Ticker"].nunique()
        num_unique_stocks_after = combined_df[combined_df["Ticker_Type"] == "Stock"][
            "Ticker"
        ].nunique()

        print(f"Number of unique coins after processing: {num_unique_coins_after}")
        print(f"Number of unique stocks after processing: {num_unique_stocks_after}")

        # Add technical indicators to the combined DataFrame
        combined_df = add_ta_indicators(combined_df)

        # Add custom calculations to the combined DataFrame
        combined_df = add_custom_calculations(combined_df)

        # Fetch market indices data and round timestamps to the nearest 30 minutes
        market_indices = ["^GSPC", "^IXIC", "^RUT", "^DJI", "^SPX", "^VIX"]
        indices_df = fetch_market_indices(market_indices)

        # Merge the market indices data with the combined DataFrame
        combined_df["Date"] = pd.to_datetime(combined_df["Date"], utc=True).dt.tz_convert(
            "Europe/Berlin"
        )
        combined_df.set_index("Date", inplace=True)
        combined_df = combined_df.join(indices_df, how="left")

        # Function to check for completely empty columns
        def check_completely_empty_columns(df):
            completely_empty_columns = df.columns[df.isnull().all()].tolist()
            return completely_empty_columns

        # Function to check for columns with only zero values
        def check_zero_only_columns(df):
            zero_only_columns = df.columns[(df == 0).all()].tolist()
            return zero_only_columns

        # Function to count unique tickers and number of non-null entries for market indices
        def ticker_and_market_index_summary(df, market_indices):
            ticker_count = df["Ticker"].nunique()
            market_index_summary = df[market_indices].count()
            return ticker_count, market_index_summary

        # Perform checks on the combined DataFrame
        completely_empty_columns = check_completely_empty_columns(combined_df)
        zero_only_columns = check_zero_only_columns(combined_df)
        ticker_count, market_index_summary = ticker_and_market_index_summary(
            combined_df, market_indices
        )
        
        # Save the combined DataFrame to a CSV file
        combined_df.to_csv('everything_data.csv')

        # Print the results of the checks
        print("Processing complete and data saved!")
        print("Columns that are completely empty:", completely_empty_columns)
        print("Columns with only zero values:", zero_only_columns)
        print("Number of unique tickers:", ticker_count)
        print("Number of non-null entries for each market index:")
        print(market_index_summary)


    if __name__ == "__main__":
        calculations()


    """## 1.2 Truncate Data"""

    # Function to reduce memory usage of a DataFrame by optimizing data types
    def reduce_mem_usage(df):
    # Calculate and print the initial memory usage of the DataFrame
        start_mem = df.memory_usage().sum() / 1024**2
        print(f'Starting memory usage: {start_mem:.2f} MB')

        # Iterate over each column in the DataFrame
        for col in df.columns:
            col_type = df[col].dtype

            # Skip the column named 'date' (assumed to be in datetime format)
            if col != 'date':
                if col_type != object:
                    c_min = df[col].min()
                    c_max = df[col].max()

                    # Optimize integer columns
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        else:
                            df[col] = df[col].astype(np.int64)
                    # Optimize float columns
                    else:
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float16)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)
                # If column is of type 'object', no type conversion is applied
                else:
                    pass

        # Calculate and print the ending memory usage of the DataFrame
        end_mem = df.memory_usage().sum() / 1024**2
        print(f'Ending memory usage: {end_mem:.2f} MB')
        # Print the percentage reduction in memory usage
        print(f'Memory reduction: {100 * (start_mem - end_mem) / start_mem:.1f}%')

        return df


    combined_df = pd.read_csv('everything_data.csv')
    # Apply memory reduction function to the DataFrame
    reduced_df = reduce_mem_usage(combined_df)

    print(reduced_df.dtypes)

    # Growth Indicators (but not future growth)
    GROWTH = [g for g in reduced_df.keys() if (g.find('Growth_')==0)&(g.find('Future')<0)]
    GROWTH

    OHLCV = ['Open','High','Low','Close','Adj Close','Close', 'Volume', 'Close_Minus_1', 'Close_Plus_1']

    CATEGORICAL = ["Ticker", "Ticker_Type", "Coin"]

    TO_PREDICT = [g for g in reduced_df.keys() if (g.find('Future')>=0)]
    TO_PREDICT

    # Function to safely compute the logarithm of a value
    def safe_log(x):
        # Return the natural logarithm of x if x is positive, otherwise return NaN
        if x > 0:
            return np.log(x)
        else:
            return np.nan

    reduced_df['Ln_Volume'] = reduced_df['Volume'].apply(safe_log)

    CUSTOM_NUMERICAL = ['Ln_Volume', 'Adj_Close_Minus_1', 'Hour', 'Month', 'Weekday', 'Adj_Close_Plus_1', '^GSPC', '^IXIC', '^RUT', '^DJI', '^SPX', '^VIX']

    TECHNICAL_INDICATORS = [
    'ADX_14',
    'ATRr_14',
    'TRIX_30_9',
    'TRIXs_30_9',
    'KAMA_10_2_30',
    'PSARl_0.02_0.2',
    'PSARs_0.02_0.2',
    'PSARaf_0.02_0.2',
    'PSARr_0.02_0.2',
    'WMA_10',
    'DMP_14',
    'DMN_14',
    'AROOND_14',
    'AROONU_14',
    'AROONOSC_14',
    'BOP',
    'CCI_14_0.015',
    'CMO_14',
    'MACD_12_26_9',
    'MACDh_12_26_9',
    'MACDs_12_26_9',
    'MOM_10',
    'PPO_12_26_9',
    'PPOh_12_26_9',
    'PPOs_12_26_9',
    'ROC_10',
    'RSI_14',
    'STOCHk_14_3_3',
    'STOCHd_14_3_3',
    'WILLR_14',
    'BBL_5_2.0',
    'BBM_5_2.0',
    'BBU_5_2.0',
    'BBB_5_2.0',
    'BBP_5_2.0',
    'DEMA_10',
    'EMA_10',
    'MIDPOINT_2',
    'MIDPRICE_2',
    'SMA_10',
    'TEMA_10',
    'WMA_10',
    'Fibonacci_0',
    'Fibonacci_23_6',
    'Fibonacci_38_2',
    'Fibonacci_50',
    'Fibonacci_61_8',
    'Fibonacci_100']

    NUMERICAL = GROWTH + TECHNICAL_INDICATORS + CUSTOM_NUMERICAL

    print(reduced_df.Ticker.nunique())
    reduced_df.tail(1)

    """## 1.2 Dummies"""

    reduced_df['Date'] = pd.to_datetime(reduced_df['Date'], errors='coerce', utc=True)

    # Define week of the month
    if 'Date' in reduced_df.columns:
        reduced_df['WoM'] = reduced_df['Date'].apply(lambda d: (d.day-1)//7 + 1)
    else:
        print("The column 'Date' is not present.")

    # Ensure 'WoM' is an integer
    if 'WoM' in reduced_df.columns:
        reduced_df['WoM'] = reduced_df['WoM'].astype(int, errors='ignore')
        reduced_df['Month_WoM'] = reduced_df['Month'] * 10 + reduced_df['WoM']
        del reduced_df['WoM']
    else:
        print("The column 'WoM' could not be created.")

    DUMMIES = []
    new_dummy = ['Month_WoM'] + CATEGORICAL
    new_dummy_variables = pd.get_dummies(reduced_df[new_dummy], dtype='int32')

    # Add Dummy-Variables to DUMMIES
    DUMMIES.extend(new_dummy_variables.keys().to_list())

    # Concat Dummy with DataFrame
    df_with_dummies = pd.concat([reduced_df, new_dummy_variables], axis=1)

    # View resulting DataFrame
    print(f"Dummie DF: {df_with_dummies.tail(1)}")

    print(df_with_dummies.dtypes)
    print(f"Index: {df_with_dummies.index}")

    # Converting Date to right Format
    df_with_dummies['Date'] = pd.to_datetime(df_with_dummies['Date'], utc=True).dt.tz_convert('Europe/Berlin')

    # Function to add black swan events to a DataFrame
    def add_black_swan_events(df):
        # List of significant black swan events with their dates
        black_swan_events = [
        # All the dates, commented out what is too far back for the DF
        # {"Event": "Chinese Central Bank Bans Financial Institutions from Bitcoin Transactions", "Date": "2013-12-05"},
        # {"Event": "MT. Gox Bancruptcy", "Date": "2014-02-24"},
        # {"Event": "DAO Hack", "Date": "2016-06-17"},
        # {"Event": "Chinese Central Bank Inspects Bitcoin Exchanges", "Date": "2017-01-05"},
        # {"Event": "China ICO Ban", "Date": "2017-09-04"},
        # {"Event": "South Korea Regulation of Cryptocurrency", "Date": "2017-12-28"},
            {"Event": "COVID-19 Market Crash", "Date": "2020-03-12"},
            {"Event": "Elon Musk halts BTC acceptance", "Date": "2021-05-12"},
            {"Event": "China Regulations on Mining Operations", "Date": "2021-05-19"},
            {"Event": "China Crypto Ban Announcement", "Date": "2021-09-20"},
            {"Event": "Omicron COVID-19 Variant Impact on USA", "Date": "2021-12-03"},
            {"Event": "Federal Crypto Regulations due to Russia Sanctions", "Date": "2022-03-02"},
            {"Event": "Luna Depagging from USDT", "Date": "2022-05-05"},
            {"Event": "Celsius Network Bankruptcy", "Date": "2022-06-13"},
            {"Event": "FTX Exchange Collapse", "Date": "2022-11-08"},
            {"Event": "MT. Gox moves BTC for Payout", "Date": "2022-05-07"},
            {"Event": "Germany moves 1300 BTC", "Date": "2022-05-07"},
        ]
        # Iterate over each black swan event
        for event in black_swan_events:
            event_date = pd.to_datetime(event['Date'], utc=True).tz_convert('Europe/Berlin')
            event_name = event['Event']
            window = pd.Timedelta(days=7)
            # Add a new column to the DataFrame indicating whether the date falls within the event window
            df[event_name] = ((df['Date'] >= (event_date - window)) & (df['Date'] <= (event_date + window))).astype(int)
        return df

    # Function to add Bitcoin halving events to a DataFrame
    def add_bitcoin_halvings(df):
        bitcoin_halvings = [
            #{"Event": "Bitcoin Halving 1", "Date": "2012-11-28"}, date too far back for df
            #{"Event": "Bitcoin Halving 2", "Date": "2016-07-09"}, date too far back for df
            {"Event": "Bitcoin Halving 3", "Date": "2020-05-11"},
            {"Event": "Bitcoin Halving 4", "Date": "2024-04-08"},
        ]
        # Iterate over each Bitcoin halving event
        for event in bitcoin_halvings:
            event_date = pd.to_datetime(event['Date'], utc=True).tz_convert('Europe/Berlin')
            event_name = event['Event']
            window = pd.Timedelta(days=7)
            # Add a new column to the DataFrame indicating whether the date falls within the event window
            df[event_name] = ((df['Date'] >= (event_date - window)) & (df['Date'] <= (event_date + window))).astype(int)
        return df

    # Treating missing values
    def treat_missing_values(df: pd.DataFrame):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    # Applying Functions to DataFrame
    df_with_dummies = add_black_swan_events(df_with_dummies)
    df_with_dummies = add_bitcoin_halvings(df_with_dummies)
    df_with_dummies = treat_missing_values(df_with_dummies)

    # Show Results
    pd.set_option('display.max_columns', None)
    print(df_with_dummies.columns)
    print(df_with_dummies.head(2))

    DUMMIES.extend(df_with_dummies.columns[df_with_dummies.columns.str.contains('COVID-19 Market Crash|Elon Musk halts BTC acceptance|China Regulations on Mining Operations|China Crypto Ban Announcement|Omicron COVID-19 Variant Impact on USA|Federal Crypto Regulations due to Russia Sanctions|Luna Depagging from USDT|Celsius Network Bankruptcy|FTX Exchange Collapse|MT. Gox moves BTC for Payout|Germany moves 1300 BTC|Bitcoin Halving 3|Bitcoin Halving 4')].tolist())

    # Add new features
    # RSI binned, with transformed float for pd.cut function
    df_with_dummies['RSI_14'] = df_with_dummies['RSI_14'].astype('float32')
    if 'RSI_14' in df_with_dummies.columns:
        df_with_dummies['RSI_binned'] = pd.cut(df_with_dummies['RSI_14'], bins=[0, 30, 70, 100], labels=['oversold', 'neutral', 'overbought'])
    else:
        print("RSI_14 column not found in DataFrame")

    # Pct Change
    if 'Close' in df_with_dummies.columns:
        df_with_dummies['Pct_Change'] = df_with_dummies['Close'].pct_change()
    else:
        print("Close column not found in DataFrame")

    # MACD and MACD binned
    if 'MACD_12_26_9' in df_with_dummies.columns and 'MACDs_12_26_9' in df_with_dummies.columns and 'MACDh_12_26_9' in df_with_dummies.columns:
        df_with_dummies['MACD_12_26_9'] = df_with_dummies['MACD_12_26_9'].astype('float32')
        df_with_dummies['MACDs_12_26_9'] = df_with_dummies['MACDs_12_26_9'].astype('float32')
        df_with_dummies['MACDh_12_26_9'] = df_with_dummies['MACDh_12_26_9'].astype('float32')
        df_with_dummies['MACD'], df_with_dummies['MACD_signal'], df_with_dummies['MACD_hist'] = df_with_dummies['MACD_12_26_9'], df_with_dummies['MACDs_12_26_9'], df_with_dummies['MACDh_12_26_9']
        df_with_dummies['MACD_binned'] = pd.cut(df_with_dummies['MACD'], bins=[-np.inf, -0.5, 0.5, np.inf], labels=['sell', 'neutral', 'buy'])
    else:
        print("MACD columns not found in DataFrame")

    # ADX binned
    if 'ADX_14' in df_with_dummies.columns:
        df_with_dummies['ADX_14'] = df_with_dummies['ADX_14'].astype('float32')
        df_with_dummies['ADX_binned'] = pd.cut(df_with_dummies['ADX_14'], bins=[0, 25, 50, 75, 100], labels=['weak', 'moderate', 'strong', 'very strong'])
    else:
        print("ADX_14 column not found in DataFrame")

    # Volume binned
    if 'Volume' in df_with_dummies.columns:
        df_with_dummies['Volume'] = df_with_dummies['Volume'].astype('float32')
        df_with_dummies['Volume_binned'] = pd.cut(df_with_dummies['Volume'], bins=[0, df_with_dummies['Volume'].quantile(0.33), df_with_dummies['Volume'].quantile(0.66), np.inf], labels=['low', 'medium', 'high'])
    else:
        print("Volume column not found in DataFrame")

    # Aroon Oscillator binned
    if 'AROONOSC_14' in df_with_dummies.columns:
        df_with_dummies['AROONOSC_14'] = df_with_dummies['AROONOSC_14'].astype('float32')
        df_with_dummies['Aroon_binned'] = pd.cut(df_with_dummies['AROONOSC_14'], bins=[-100, -50, 50, 100], labels=['downtrend', 'neutral', 'uptrend'])
    else:
        print("AROONOSC_14 column not found in DataFrame")

    # Bollinger Bands Percentage binned
    if 'BBP_5_2.0' in df_with_dummies.columns:
        df_with_dummies['BBP_5_2.0'] = df_with_dummies['BBP_5_2.0'].astype('float32')
        df_with_dummies['BBP_binned'] = pd.cut(df_with_dummies['BBP_5_2.0'], bins=[0, 0.2, 0.8, 1], labels=['low', 'medium', 'high'])
    else:
        print("BBP_5_2.0 column not found in DataFrame")

    # SMA cross
    if 'Close' in df_with_dummies.columns:
        df_with_dummies['SMA_50'] = df_with_dummies['Close'].rolling(window=50).mean()
        df_with_dummies['SMA_200'] = df_with_dummies['Close'].rolling(window=200).mean()
        df_with_dummies['SMA_Cross'] = df_with_dummies['SMA_50'] - df_with_dummies['SMA_200']
    else:
        print("Close column not found in DataFrame")

    # Drop rows with NaN values in new features if necessary
    new_features = ['Pct_Change', 'SMA_Cross']# you see the all features printed before binning, this was corrected hier afterwards
    df_with_dummies.dropna(subset=new_features, inplace=True)

    # Generate dummy variables for the new binned features
    new_categorical = ['RSI_binned', 'MACD_binned', 'ADX_binned', 'Volume_binned', 'Aroon_binned', 'BBP_binned']
    new_dummy_variables = pd.get_dummies(df_with_dummies[new_categorical], dtype='int32')

    # Concatenate the new dummy variables with the original DataFrame
    df_with_all_dummies = pd.concat([df_with_dummies, new_dummy_variables], axis=1)

    df_with_all_dummies = df_with_all_dummies.reset_index(drop=True)

    binned_features = [
            "RSI_binned",
            "MACD_binned",
            "ADX_binned",
            "Volume_binned",
            "Aroon_binned",
            "BBP_binned",
        ]

    TO_DROP = ["Close"] + binned_features + ["Volume"] +  ["Coin"]
    df_with_all_dummies.drop(columns=TO_DROP, inplace=True, errors="ignore")
    print(f"Values to drop: {TO_DROP}")

    # CHECK: NO OTHER INDICATORS LEFT
    OTHER = [k for k in reduced_df.keys() if k not in OHLCV + CATEGORICAL + NUMERICAL + TO_DROP + TO_PREDICT]
    print(f"Left-over columns, should be none :{OTHER}")


    pd.set_option('display.max_columns', None)
    print(df_with_all_dummies.head(2))

    NUMERICAL.extend([
        'Pct_Change',
        'SMA_50',
        'SMA_200',
        'SMA_Cross'
    ])

    DUMMIES.extend([
        'RSI_binned_oversold',
        'RSI_binned_neutral',
        'RSI_binned_overbought',
        'MACD_binned_sell',
        'MACD_binned_neutral',
        'MACD_binned_buy',
        'ADX_binned_weak',
        'ADX_binned_moderate',
        'ADX_binned_strong',
        'ADX_binned_very strong',
        'Volume_binned_low',
        'Volume_binned_medium',
        'Volume_binned_high',
        'Aroon_binned_downtrend',
        'Aroon_binned_neutral',
        'Aroon_binned_uptrend',
        'BBP_binned_low',
        'BBP_binned_medium',
        'BBP_binned_high'
    ])

    print(f"List of numerical features: {NUMERICAL}")

    print(f"List of created dummies:{DUMMIES}")

    with open("column_lists.pkl", "wb") as f:
        pickle.dump(
                {
                    "GROWTH": GROWTH,
                    "TO_PREDICT": TO_PREDICT,
                    "CUSTOM_NUMERICAL": CUSTOM_NUMERICAL,
                    "TECHNICAL_INDICATORS": TECHNICAL_INDICATORS,
                    "NUMERICAL": NUMERICAL,
                    "DUMMIES": DUMMIES,
                },
                f,
            )


    df_with_all_dummies.to_csv("df_with_all_dummies.csv", index=False)

if __name__ == "__main__":
    get_data()