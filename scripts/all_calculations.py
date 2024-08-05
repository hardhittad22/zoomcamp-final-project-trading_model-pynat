import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import time

# Function to calculate technical indicators
def add_ta_indicators(df):
    df['Open'] = df['Open'].astype('float64')
    df['High'] = df['High'].astype('float64')
    df['Low'] = df['Low'].astype('float64')
    df['Close'] = df['Close'].astype('float64')
    df['Volume'] = df['Volume'].astype('float64')

    # Calculating Indicators relevant for swing trading
    df.ta.adx(high='High', low='Low', close='Close', append=True)
    df.ta.aroon(high='High', low='Low', append=True)
    df.ta.bop(open='Open', high='High', low='Low', close='Close', append=True)
    df.ta.cci(high='High', low='Low', close='Close', append=True)
    df.ta.cmo(close='Close', append=True)
    df.ta.macd(close='Close', append=True)
    df.ta.mfi(high='High', low='Low', close='Close', volume='Volume', append=True)
    df.ta.mom(close='Close', append=True)
    df.ta.ppo(close='Close', append=True)
    df.ta.roc(close='Close', append=True)
    df.ta.rsi(close='Close', append=True)
    df.ta.stoch(high='High', low='Low', close='Close', append=True)
    df.ta.willr(high='High', low='Low', close='Close', append=True)
    df.ta.bbands(close='Close', append=True)
    df.ta.dema(close='Close', append=True)
    df.ta.ema(close='Close', append=True)
    df.ta.midpoint(close='Close', append=True)
    df.ta.midprice(high='High', low='Low', append=True)
    df.ta.sma(close='Close', append=True)
    df.ta.tema(close='Close', append=True)
    df.ta.wma(close='Close', append=True)
    df.ta.atr(high='High', low='Low', close='Close', append=True)
    df.ta.trix(close='Close', append=True)
    df.ta.kama(close='Close', append=True)
    df.ta.psar(high='High', low='Low', close='Close', append=True)

    return df

# Function to check for empty columns
def check_empty_columns(df):
    empty_columns = df.columns[df.isnull().any()].tolist()
    return empty_columns

# Function to check for completely empty columns
def check_completely_empty_columns(df):
    completely_empty_columns = df.columns[df.isnull().all()].tolist()
    return completely_empty_columns

# Function to check for columns with only zero values
def check_zero_only_columns(df):
    zero_only_columns = df.columns[(df == 0).all()].tolist()
    return zero_only_columns

# Function to clean duplicate column names
def clean_duplicate_columns(df):
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# Function to add shifted and growth columns
def add_shifted_and_growth_columns(df):
    group_by_col = 'Ticker'
    
     # Create a dictionary for new columns
    new_columns = {}

    new_columns['Adj_Close_Minus_1'] = df.groupby(group_by_col)['Close'].shift(-1)
    new_columns['Adj_Close_Plus_1'] = df.groupby(group_by_col)['Close'].shift(1)

    # Calculation for different time periods
    for hours in [1, 4, 24, 48, 72, 168, 336, 720]:
        new_columns[f'Growth_{hours}h'] = df.groupby(group_by_col)['Close'].shift(hours) / df['Close']
        future_shifted = df.groupby(group_by_col)['Close'].shift(-hours)
        new_columns[f'Growth_Future_{hours}h'] = future_shifted / df['Close']
        new_columns[f'Is_Positive_Growth_{hours}h_Future'] = np.where(new_columns[f'Growth_Future_{hours}h'] > 1, 1, 0)

    # Add all new columns at once
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    return df

# Function to add Fibonacci levels
def add_fibonacci_levels(df):
    new_columns = {
        'Fibonacci_0': df['Adj Close'],
        'Fibonacci_23_6': df['High'] - (df['High'] - df['Low']) * 0.236,
        'Fibonacci_38_2': df['High'] - (df['High'] - df['Low']) * 0.382,
        'Fibonacci_50': df['High'] - (df['High'] - df['Low']) * 0.5,
        'Fibonacci_61_8': df['High'] - (df['High'] - df['Low']) * 0.618,
        'Fibonacci_100': df['Low']
    }

    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df

# Function to fetch index data
def get_index_data(index, period='2y', interval='1h'):
    data = yf.download(index, period=period, interval=interval)
    data.index = data.index.tz_convert('Europe/Berlin')  # Timezone convert
    data['Date'] = data.index
    return data[['Date', 'Close']].rename(columns={'Close': f'Close_{index}'})

def remove_duplicate_timestamps(df):
    df = df[~df.index.duplicated(keep='first')]
    return df

def calculations():
    # Read Data
    crypto_df = pd.read_csv('crypto.csv', index_col='Date', parse_dates=True)
    stocks_df = pd.read_csv('stocks.csv', index_col='Date', parse_dates=True)

    # Timezones for DFs
    crypto_df.index = pd.to_datetime(crypto_df.index, utc=True).tz_convert('Europe/Berlin')
    stocks_df.index = pd.to_datetime(stocks_df.index, utc=True).tz_convert('Europe/Berlin')

    # Defining Market Indices
    market_indices = ['^GSPC', '^IXIC', '^RUT', '^DJI', '^SPX', '^VIX']

    # Calling Indices
    indices_data = {}
    for index in market_indices:
        print(f'Downloading stats for index: {index}')
        indices_data[index] = get_index_data(index)
        time.sleep(0.5)  # Pause to avoid API limits

    # Adding Indizes
    for index, index_data in indices_data.items():
        index_data = index_data.reset_index(drop=True)
        crypto_df = pd.merge(crypto_df.reset_index(), index_data, on='Date', how='left').set_index('Date')
        stocks_df = pd.merge(stocks_df.reset_index(), index_data, on='Date', how='left').set_index('Date')

    # Clean duplicate column names
    stocks_df = clean_duplicate_columns(stocks_df)
    crypto_df = clean_duplicate_columns(crypto_df)

    # Merge DataFrames
    combined_df = pd.concat([crypto_df, stocks_df], ignore_index=False)

    # Remove duplicate timestamps
    combined_df = remove_duplicate_timestamps(combined_df)

    # Apply functions
    combined_df = add_shifted_and_growth_columns(combined_df)
    combined_df = add_fibonacci_levels(combined_df)

    # Add technical indicators
    combined_df = add_ta_indicators(combined_df)

    # Check for empty or zero-value columns
    empty_columns = check_empty_columns(combined_df)
    completely_empty_columns = check_completely_empty_columns(combined_df)
    zero_only_columns = check_zero_only_columns(combined_df)

    print("Columns that are empty:", empty_columns)
    print("Columns that are completely empty:", completely_empty_columns)
    print("Columns with just Zero:", zero_only_columns)

    # Save combined DataFrame
    combined_df.to_csv('everything_data.csv', index=True)

    # Display results 
    print("crypto_df_filtered head:", crypto_df.head(2))
    print("stocks_df head:", stocks_df.head(2))
    print("combined_df head:", combined_df.head(2))

    print("crypto_df_filtered index type:", type(crypto_df.index))
    print("stocks_df index type:", type(stocks_df.index))

    print("crypto_df_filtered Date range:")
    print(crypto_df.index.min(), crypto_df.index.max())
    print("stocks_df Date range:")
    print(stocks_df.index.min(), stocks_df.index.max())

    for index, index_data in indices_data.items():
        print(f'{index} Date range:')
        print(index_data.index.min(), index_data.index.max())

# Main Funktion
if __name__ == "__main__":
    calculations()
