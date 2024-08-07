import pandas as pd
import yfinance as yf
import time

# Function to download historical stock data
def get_stats(ticker, period='2y', interval='1h'):
    data = yf.download(ticker, period=period, interval=interval)
    return data

# Adding Date and Time Features
def add_date_features_and_shifts(df):
    # Convert index to datetime and localize to Berlin timezone
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert('Europe/Berlin')
    df.set_index('Date', inplace=True)
    # Extract year, month, weekday, and hour from the datetime index
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Weekday'] = df.index.weekday
    df['Hour'] = df.index.hour

def top_stocks():
    # Getting top stocks from yahoofinance
    yfinance_most_active_stocks_web = pd.read_html("https://finance.yahoo.com/most-active/?offset=0&count=100")
    FIELDS = ['Symbol', 'Name', 'Volume']
    stock_tickers = yfinance_most_active_stocks_web[0][FIELDS]

    # Initialize an empty DataFrame for storing stock data
    stocks_df = None
    period = '2y'
    interval = '1h'

    # Loop through each stock ticker and download data
    for elem in stock_tickers.Symbol.to_list():
        print(f'Downloading stats for stock: {elem}')
        one_ticker_df = get_stats(elem, period=period, interval=interval)
        if one_ticker_df.empty:
            print(f'No data available for stock: {elem}')
            continue
        # Add additional columns to the DataFrame
        one_ticker_df['Date'] = one_ticker_df.index
        one_ticker_df['Ticker'] = elem
        one_ticker_df['Ticker_Type'] = 'Stock'
        # Concatenate each stock's data into a single DataFrame
        if stocks_df is None:
            stocks_df = one_ticker_df
        else:
            stocks_df = pd.concat([stocks_df, one_ticker_df], ignore_index=True, axis=0)
        time.sleep(0.5)  # pause to avoid API overload

    # Adding date and time features
    add_date_features_and_shifts(stocks_df)

    # Reorder columns to have date and time features at the beginning
    cols = ['Year', 'Month', 'Weekday', 'Hour', 'Ticker', 'Ticker_Type'] + [col for col in stocks_df.columns if col not in ['Year', 'Month', 'Weekday', 'Hour', 'Ticker', 'Ticker_Type']]
    stocks_df = stocks_df[cols]

    # save DataFrame as csv
    stocks_df.to_csv('stocks.csv', index=True)

    # Print the first few rows
    print(stocks_df.head())

if __name__ == "__main__":
    top_stocks()
