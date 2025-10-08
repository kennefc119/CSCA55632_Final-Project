import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
import csv

# Function to fetch and clean stock data using yfinance
def fetch_clean_data(ticker_symbol, period="120d", interval="1d"):
    # Fetch Data from Yfinance
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period, interval=interval)
    
    #Clean Data
    data.dropna(inplace=True)
    data.interpolate(method='linear', inplace=True)

    # Remove Duplicates
    data = data[~data.index.duplicated(keep='first')]
    data = data.astype(float)
    
    # Reset Index and Convert Date
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])

    return data 

# Function to sort DataFrame by Date
def sort_by_date(df):
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df.sort_values('Date').reset_index(drop=True)

# Function to get all US equity tickers from NASDAQ
def get_us_equity_tickers():
    urls = [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol")
    ]

    tickers = []

    for url, symbol_col in urls:
        r = requests.get(url)
        content = r.text.split("File Creation Time")[0]
        df = pd.read_csv(StringIO(content), sep="|")
        
        #Filter out ETFs and Test Issues if those columns exist
        if "ETF" in df.columns:
            df = df[df["ETF"] != "Y"]

        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] != "Y"]

        tickers.extend(df[symbol_col].tolist())

    return list(set(tickers))

# Function to clean tickers based on specific criteria
def clean_lid_symbol(symbol):
    if not isinstance(symbol, str):
        return False
    return (
        "$" not in symbol and
        "^" not in symbol and
        "=" not in symbol and
        ".U" not in symbol and
        ".W" not in symbol and
        "." not in symbol and
        ".P" not in symbol
    )



# Convert all columns except 'Ticker' to float
def clean_ticker_matrix(df):
    df.update(df.drop(columns="Ticker").astype(float))
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    return df


# Main function to compute features over sliding windows
def compute_sliding_window_features(df, shift_days, partition_days):
    total_length = len(df)
    
    df_parts = []
    for i in range(int(total_length/shift_days)):
        part = df[i*shift_days:i*shift_days+partition_days]
        if (len(part)) != partition_days:
            continue
        df_parts.append(part)
    
    combined_result = {}
    for i,df_concerned in enumerate(df_parts):
        combined_result.update(derive_relative_change_features(df_concerned, i+1))
        combined_result.update(compute_candle_ratio(df_concerned, i+1))

    return pd.DataFrame([combined_result])

# Function to derive relative change features
def derive_relative_change_features(df_concerned, part_ID):
   result = {}
   close = df_concerned['Close']
   open = df_concerned['Open']
   high = df_concerned['High']
   low = df_concerned['Low']
   volume = df_concerned['Volume']
   length = df_concerned.shape[0]

   # Overall Metric over the entire window
   names = ['close', 'open', 'high', 'low','volume']
   price_data = [close, open, high, low, volume]
   for name,data in zip(names, price_data):
    mean = data.mean()
    min = data.min()
    max = data.max()
    result[f'price_overall_{name}_change_portion_{part_ID}'] = (data.iloc[-1] - data.iloc[0]) / data.iloc[0]
    result[f'price_overall_{name}_relative_max_portion_{part_ID}'] = abs(max-mean)/mean
    result[f'price_overall_{name}_relative_min_portion_{part_ID}'] = abs(min-mean)/mean

    # Pairwise metric every 2 days
    for name, data in zip(names, price_data):
        pairwise_percent_change = []
        for i in range(length-1):
            change = (data.iloc[i + 1] - data.iloc[i]) / data.iloc[i]
            pairwise_percent_change.append(change)
        if len(pairwise_percent_change) != 0:
            result[f'price_pairwise_{name}_mean_portion_{part_ID}'] = sum(pairwise_percent_change) / len(pairwise_percent_change)
        else:
            result[f'price_pairwise_{name}_mean_portion_{part_ID}'] = 0.0

   return result

# Function to compute candle ratio features
def compute_candle_ratio(df_concerned, part_ID):
    result = {}
    close = df_concerned['Close']
    open = df_concerned['Open']
    high = df_concerned['High']
    low = df_concerned['Low']
    length = df_concerned.shape[0]
    
    body_size = abs(close - open)
    full_range = high - low
    body_to_range_ratio = body_size / full_range
    result[f'candle_body_ratio_portion_{part_ID}'] = body_to_range_ratio.mean()
    
    # Pairwise Candle Body Change
    pairwise_body_change = []
    for i in range(length - 1):
        prev = body_size.iloc[i]
        curr = body_size.iloc[i+1]
        if prev != 0:
            change = (curr - prev) / prev
            pairwise_body_change.append(change)
    if pairwise_body_change:
        result[f'candle_pairwise_body_change_mean_{part_ID}'] = sum(pairwise_body_change) / len(pairwise_body_change)

    return result


# Create output CSV and write header
def ticker_matrix_creation(tickers,analysis_days_split,_step_shift_days,_partition_days,_period= 120,_interval = '1d',local_run = False,output_file = "data/ticker_matrix.csv"):
    _period = str(_period+10) + 'd'
    open(output_file, mode='w').close()
    header_written = False
    
    # Iterate through each ticker, fetch data, compute features, and write to CSV
    for ticker in tickers:
        if local_run == False:
            df = fetch_clean_data(ticker, period=_period, interval=_interval)
            df = sort_by_date(df)
            path = f"data/{ticker}.csv"
            df.to_csv(path)
        else:
            try:
                df = pd.read_csv(f"data/{ticker}.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d %H:%M:%S%z", nrows=_period)
                df = df[:int(_period.split('d')[0])]
            except:
                continue
        
        # Compute features with error handling for invalid operations
        with np.errstate(invalid='ignore', divide='ignore'):
            ticker_df = compute_sliding_window_features(df, shift_days = _step_shift_days, partition_days = _partition_days)
        
        headers = ['Ticker']  + ticker_df.columns.tolist()
        vector = ticker_df.values[0]
        
        # Write to CSV
        with open(output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if header_written == False:
                writer.writerow(headers)
                header_written = True
            writer.writerow([ticker] + vector.tolist())
