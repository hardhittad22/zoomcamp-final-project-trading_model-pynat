import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import time


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
        df["Year"] = df.index.year
        df["Month"] = df.index.month
        df["Weekday"] = df.index.weekday
        df["Hour"] = df.index.hour
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
    all_data.to_csv("crypto.csv", index=True)
    logging.info("Data fetching and processing complete. Data saved to crypto.csv")


if __name__ == "__main__":
    get_coins()
