import pandas as pd
import yfinance as yf
import numpy as np
import pandas_ta as ta


# Ensure Date is datetime and handle timezone conversions correctly
def ensure_datetime_with_tz(df, date_col="Date", target_tz="Europe/Berlin"):
    # Convert to datetime and force UTC timezone awareness
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")

    # Convert to target timezone
    df[date_col] = df[date_col].dt.tz_convert(target_tz)

    return df


def add_ta_indicators(df):
    # Ensure the columns are in the correct data types
    df["Open"] = df["Open"].astype("float64")
    df["High"] = df["High"].astype("float64")
    df["Low"] = df["Low"].astype("float64")
    df["Close"] = df["Close"].astype("float64")
    df["Volume"] = df["Volume"].astype("float64")

    # Calculate various technical indicators relevant for swing trading
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


def add_custom_calculations(df, group_by_col="Ticker"):
    new_columns = {}

    # Shift the 'Close' values forward and backward
    new_columns["Adj_Close_Minus_1"] = df.groupby(group_by_col)["Close"].shift(-1)
    new_columns["Adj_Close_Plus_1"] = df.groupby(group_by_col)["Close"].shift(1)

    # Calculate growth over different time periods
    for hours in [1, 4, 24, 48, 72, 168, 336, 720]:
        new_columns[f"Growth_{hours}h"] = (
            df.groupby(group_by_col)["Close"].shift(hours) / df["Close"]
        )
        future_shifted = df.groupby(group_by_col)["Close"].shift(-hours)
        new_columns[f"Growth_Future_{hours}h"] = future_shifted / df["Close"]
        new_columns[f"Is_Positive_Growth_{hours}h_Future"] = np.where(
            new_columns[f"Growth_Future_{hours}h"] > 1, 1, 0
        )

    # Calculate Fibonacci levels
    new_columns["Fibonacci_0"] = df["Adj Close"]
    new_columns["Fibonacci_23_6"] = df["High"] - (df["High"] - df["Low"]) * 0.236
    new_columns["Fibonacci_38_2"] = df["High"] - (df["High"] - df["Low"]) * 0.382
    new_columns["Fibonacci_50"] = df["High"] - (df["High"] - df["Low"]) * 0.5
    new_columns["Fibonacci_61_8"] = df["High"] - (df["High"] - df["Low"]) * 0.618
    new_columns["Fibonacci_100"] = df["Low"]

    # Add the new columns to the DataFrame
    df = df.assign(**new_columns)

    return df


# Function to check for completely empty columns
def check_completely_empty_columns(df):
    completely_empty_columns = df.columns[df.isnull().all()].tolist()
    return completely_empty_columns


# Function to check for columns with only zero values
def check_zero_only_columns(df):
    zero_only_columns = df.columns[(df == 0).all()].tolist()
    return zero_only_columns


def fetch_market_indices(market_indices):
    # Dictionary to store the index data
    indices_data = {}

    # Fetch data for each market index
    for index in market_indices:
        data = yf.download(index, period="1y", interval="1h")
        data = data[["Adj Close"]].rename(columns={"Adj Close": index})
        indices_data[index] = data

    # Combine all indices data into one DataFrame
    combined_indices = pd.concat(indices_data.values(), axis=1)

    return combined_indices


def calculations():
    # Fetch market data
    market_indices = ["^GSPC", "^IXIC", "^RUT", "^DJI", "^SPX", "^VIX"]
    indices_df = fetch_market_indices(market_indices)
    indices_df.index = indices_df.index.tz_convert("Europe/Berlin")
    # Round the timestamps of 'indices_df' to the nearest hour
    indices_df.index = indices_df.index.round("h")

    # Load the existing DataFrames
    crypto_df = pd.read_csv("crypto.csv")
    stocks_df = pd.read_csv("stocks.csv")

    # Combine the crypto and stock DataFrames
    one_df = pd.concat([crypto_df, stocks_df], ignore_index=True)

    # Ensure Date column is properly formatted with timezone awareness
    one_df = ensure_datetime_with_tz(one_df)

    # Add technical indicators
    one_df = add_ta_indicators(one_df)

    # Add custom calculations
    one_df = add_custom_calculations(one_df)

    # Set 'Date' as the index and ensure no duplicates
    one_df.set_index("Date", inplace=True)
    one_df = one_df[~one_df.index.duplicated(keep="first")]
    indices_df = indices_df[~indices_df.index.duplicated(keep="first")]

    # make sure they use the same index
    one_df.index = pd.to_datetime(one_df.index)
    indices_df.index = pd.to_datetime(indices_df.index)

    completely_empty_columns = check_completely_empty_columns(one_df)
    zero_only_columns = check_zero_only_columns(one_df)

    combined_df = pd.concat([one_df, indices_df], axis=1)

    # Savig to csv DataFrames
    combined_df.to_csv("everything_data.csv", index=False)

    print("Market indices successfully added and saved!")
    print("Columns that are completely empty:", completely_empty_columns)
    print("Columns with just Zero:", zero_only_columns)

    print("combined_df head:", one_df.head(2))


if __name__ == "__main__":
    calculations()
