import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta


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


# Function to safely compute the logarithm of a value
def safe_log(x):
    # Return the natural logarithm of x if x is positive, otherwise return NaN
    if x > 0:
        return np.log(x)
    else:
        return np.nan

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

def prepare():
    # Load the dataset from a CSV file
    combined_df = pd.read_csv('everything_data.csv')

    reduced_df = reduce_mem_usage(combined_df)
    reduced_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    GROWTH = [g for g in reduced_df.keys() if (g.find('Growth_') == 0) & (g.find('Future') < 0)]
    OHLCV = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Close', 'Volume', 'Close_Minus_1', 'Close_Plus_1']
    TO_PREDICT = [g for g in reduced_df.keys() if (g.find('Future') >= 0)]
    reduced_df['Ln_Volume'] = reduced_df['Volume'].apply(safe_log)
    CUSTOM_NUMERICAL = ['Ln_Volume', 'Adj_Close_Minus_1', 'Adj_Close_Plus_1', 'Hour', 'Month', 'Close_^GSPC', 'Close_^IXIC', 'Close_^RUT', 'Close_^DJI', 'Close_^SPX', 'Close_^VIX']
    TECHNICAL_INDICATORS = [
        'ADX_14', 'ATRr_14', 'TRIX_30_9', 'TRIXs_30_9', 'KAMA_10_2_30', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2',
        'PSARr_0.02_0.2', 'WMA_10', 'DMP_14', 'DMN_14', 'AROOND_14', 'AROONU_14', 'AROONOSC_14', 'BOP', 'CCI_14_0.015', 'CMO_14',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'MFI_14', 'MOM_10', 'PPO_12_26_9', 'PPOh_12_26_9', 'PPOs_12_26_9',
        'ROC_10', 'RSI_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'WILLR_14', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0',
        'BBP_5_2.0', 'DEMA_10', 'EMA_10', 'MIDPOINT_2', 'MIDPRICE_2', 'SMA_10', 'TEMA_10', 'WMA_10', 'Fibonacci_0', 'Fibonacci_23_6',
        'Fibonacci_38_2', 'Fibonacci_50', 'Fibonacci_61_8', 'Fibonacci_100'
    ]
    CATEGORICAL = ['Ticker', 'Ticker_Type']
    TO_DROP = ['Year','Date'] + CATEGORICAL + OHLCV + ['Coin'] + ['Unnamed: 0']
    NUMERICAL = GROWTH + TECHNICAL_INDICATORS + CUSTOM_NUMERICAL

    OTHER = [k for k in reduced_df.keys() if k not in OHLCV + TO_DROP + CATEGORICAL + NUMERICAL + TO_PREDICT]
    print(OTHER)
    print(reduced_df.Ticker.nunique())
    reduced_df.tail(1)

    DUMMIES = ['Month', 'Weekday', 'Month_WoM']

    # Ensure 'Date' column is present and convert it to a consistent date format
    if 'Date' in reduced_df.columns:
        reduced_df['Date'] = pd.to_datetime(reduced_df['Date'], errors='coerce')
        reduced_df.dropna(subset=['Date'], inplace=True)
    else:
        print("The column 'Date' is not present.")

    # Convert 'Month' and 'Weekday' columns to integers if they exist
    if 'Month' in reduced_df.columns:
        reduced_df['Month'] = reduced_df['Month'].astype(int, errors='ignore')
    else:
        print("The column 'Month' is not present.")

    if 'Weekday' in reduced_df.columns:
        reduced_df['Weekday'] = reduced_df['Weekday'].astype(int, errors='ignore')
    else:
        print("The column 'Weekday' is not present.")

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

    # Generate dummy variables
    dummy_variables = pd.get_dummies(reduced_df[CATEGORICAL], dtype='int32')

    # Concatenate the dummy variables with the original DataFrame
    df_with_dummies = pd.concat([reduced_df, dummy_variables], axis=1)

    # Converte Date
    df_with_dummies['Date'] = pd.to_datetime(df_with_dummies['Date'], utc=True).dt.tz_convert('Europe/Berlin')

    # Apply Functions
    df_with_dummies = add_black_swan_events(df_with_dummies)
    df_with_dummies = add_bitcoin_halvings(df_with_dummies)
    df_with_dummies = treat_missing_values(df_with_dummies)

    # Show Results
    pd.set_option('display.max_columns', None)
    print(df_with_dummies.columns)
    print(df_with_dummies.head(2))

    # Adding new Relevant Features For Swing Trading
    # RSI binned, with transformed float for the pd.cut funktion
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
    new_features = ['Pct_Change', 'SMA_Cross']
    df_with_dummies.dropna(subset=new_features, inplace=True)

    # Generate dummy variables for the new binned features
    new_categorical = ['RSI_binned', 'MACD_binned', 'ADX_binned', 'Volume_binned', 'Aroon_binned', 'BBP_binned']
    new_dummy_variables = pd.get_dummies(df_with_dummies[new_categorical], dtype='int32')

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

    with open('column_lists.pkl', 'wb') as f:
        pickle.dump({
            'OHLCV': OHLCV,
            'GROWTH': GROWTH,
            'TO_PREDICT': TO_PREDICT,
            'CUSTOM_NUMERICAL': CUSTOM_NUMERICAL,
            'TECHNICAL_INDICATORS': TECHNICAL_INDICATORS,
            'TO_DROP': TO_DROP,
            'NUMERICAL': NUMERICAL,
            'CATEGORICAL': CATEGORICAL,
            'DUMMIES': DUMMIES
        }, f)
# Concatenate the new dummy variables with the original DataFrame
    df_with_all_dummies = pd.concat([df_with_dummies, new_dummy_variables], axis=1)

    df_with_all_dummies = df_with_all_dummies.reset_index(drop=True)

    pd.set_option('display.max_columns', None)
    print(df_with_all_dummies.head(2))

    df_with_all_dummies.to_csv('df_with_all_dummies.csv', index=False)

if __name__ == "__main__":
    prepare()
