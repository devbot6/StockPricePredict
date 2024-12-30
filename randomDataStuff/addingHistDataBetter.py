import requests
import pandas as pd
from datetime import datetime, timedelta

# API Keys
API_KEY = ""
SECRET_KEY = ""

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}

# Define parameters
symbol = 'TSLA'
start_date = "2024-09-01T00:00:00Z"  # ISO format
end_date = "2024-12-01T00:00:00Z"  # ISO format
limit = 1000  # Maximum records per request

# Fetch Historical Trades
trade_url = f"https://data.alpaca.markets/v2/stocks/trades?symbols={symbol}&start={start_date}&end={end_date}&limit={limit}"
trade_response = requests.get(trade_url, headers=headers)
if trade_response.status_code != 200:
    print(f"Error fetching trade data: {trade_response.status_code}")
else:
    trade_data = trade_response.json().get('trades', {}).get(symbol, [])

# Fetch Historical Bars
bar_url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={symbol}&timeframe=1Hour&start={start_date}&end={end_date}&limit={limit}&adjustment=all"
bar_response = requests.get(bar_url, headers=headers)
if bar_response.status_code != 200:
    print(f"Error fetching bar data: {bar_response.status_code}")
else:
    bar_data = bar_response.json().get('bars', {}).get(symbol, [])

# Fetch Historical Quotes
quote_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes?start={start_date}&end={end_date}&limit={limit}"
quote_response = requests.get(quote_url, headers=headers)
if quote_response.status_code != 200:
    print(f"Error fetching quote data: {quote_response.status_code}")
else:
    quote_data = quote_response.json().get('quotes', [])

# Check if data is empty
if not trade_data:
    print("No trade data available")
if not bar_data:
    print("No bar data available")
if not quote_data:
    print("No quote data available")

# Convert Data to DataFrames
trade_df = pd.DataFrame.from_records(
    [
        {"timestamp": trade["t"], "trade_price": trade["p"], "trade_size": trade["s"]}
        for trade in trade_data
    ]
)

bar_df = pd.DataFrame.from_records(
    [
        {"timestamp": bar["t"], "bar_open": bar["o"], "bar_high": bar["h"], "bar_low": bar["l"], "bar_close": bar["c"], "volume": bar["v"]}
        for bar in bar_data
    ]
)

quote_df = pd.DataFrame.from_records(
    [
        {"timestamp": quote["t"], "quote_bid_price": quote["bp"], "quote_ask_price": quote["ap"]}
        for quote in quote_data
    ]
)

# Merge DataFrames
merged_df = pd.merge(trade_df, bar_df, on="timestamp", how="outer")
merged_df = pd.merge(merged_df, quote_df, on="timestamp", how="outer")

# Append to CSV
file_name = "historical_stock_data2.csv"
try:
    merged_df.to_csv(file_name, mode='a', index=False, header=not pd.io.common.file_exists(file_name))
    print("Historical data appended to CSV successfully!")
except Exception as e:
    print(f"Error appending data: {e}")

# Display the DataFrame
print(merged_df)
