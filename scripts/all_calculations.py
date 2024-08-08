import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf


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
    crypto_df = pd.read_csv("crypto.csv")
    stocks_df = pd.read_csv("stocks.csv")

    unique_coins_before = crypto_df["Ticker"].nunique()
    unique_stocks_before = stocks_df["Ticker"].nunique()
    print(f"Anzahl der einzigartigen Coins vor der Verarbeitung: {unique_coins_before}")
    print(
        f"Anzahl der einzigartigen Stocks vor der Verarbeitung: {unique_stocks_before}"
    )

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
    combined_df.to_csv("everything_data.csv")

    # Print the results of the checks
    print("Processing complete and data saved!")
    print("Columns that are completely empty:", completely_empty_columns)
    print("Columns with only zero values:", zero_only_columns)
    print("Number of unique tickers:", ticker_count)
    print("Number of non-null entries for each market index:")
    print(market_index_summary)


if __name__ == "__main__":
    calculations()
