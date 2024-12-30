import requests
import pandas as pd
from datetime import datetime, timedelta

# API Keys
API_KEY = "PKCS6JK0RLPEEOCA11KO"
SECRET_KEY = "BZ9qKKrhl1wnuGxbUDiL186jPUQGg8SPwW24ikSP"

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}

# Define parameters
symbol = 'TSLA'
start_date = "2024-10-01T00:00:00Z"  # ISO format
end_date = "2024-12-01T00:00:00Z"  # ISO format
limit = 10  # Maximum records per request

# Fetch Historical Trades
trade_url = f"https://data.alpaca.markets/v2/stocks/trades?symbols={symbol}&start={start_date}&end={end_date}&limit={limit}"
trade_response = requests.get(trade_url, headers=headers).json()

# Fetch Historical Bars
bar_url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={symbol}&timeframe=1Hour&start={start_date}&end={end_date}&limit={limit}&adjustment=all"
bar_response = requests.get(bar_url, headers=headers).json()

# Fetch Historical Quotes
quote_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes?start={start_date}&end={end_date}&limit={limit}"
quote_response = requests.get(quote_url, headers=headers).json()

# Extract Data for DataFrames
trade_data = trade_response['trades'][symbol]
bar_data = bar_response['bars'][symbol]
quote_data = quote_response['quotes']

# Convert Trade Data to DataFrame
trade_df = pd.DataFrame.from_records(
    [
        {
            "timestamp": trade["t"],
            "trade_price": trade["p"],
            "trade_size": trade["s"],
        }
        for trade in trade_data
    ]
)

# Convert Bar Data to DataFrame
bar_df = pd.DataFrame.from_records(
    [
        {
            "timestamp": bar["t"],
            "bar_open": bar["o"],
            "bar_high": bar["h"],
            "bar_low": bar["l"],
            "bar_close": bar["c"],
            "volume": bar["v"],
        }
        for bar in bar_data
    ]
)

# Convert Quote Data to DataFrame
quote_df = pd.DataFrame.from_records(
    [
        {
            "timestamp": quote["t"],
            "quote_bid_price": quote["bp"],
            "quote_ask_price": quote["ap"],
        }
        for quote in quote_data
    ]
)

# Merge all data into a single DataFrame (optional)
merged_df = pd.merge(trade_df, bar_df, on="timestamp", how="outer")
merged_df = pd.merge(merged_df, quote_df, on="timestamp", how="outer")

# Append to CSV
file_name = "historical_stock_data2.csv"
try:
    # Append data to the CSV file if it exists; write the header only if the file doesn't exist
    merged_df.to_csv(file_name, mode='a', index=False, header=not pd.io.common.file_exists(file_name))
    print("Historical data appended to CSV successfully!")
except Exception as e:
    print(f"Error appending data: {e}")

# Display the DataFrame
print(merged_df)
