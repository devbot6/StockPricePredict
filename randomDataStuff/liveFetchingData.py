import requests
import pandas as pd

# API Keys
API_KEY = "PKCS6JK0RLPEEOCA11KO"
SECRET_KEY = "BZ9qKKrhl1wnuGxbUDiL186jPUQGg8SPwW24ikSP"

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}

import time

# Initialize empty DataFrame
df = pd.DataFrame(columns=["timestamp", "trade_price", "trade_size", "bar_open", "bar_high", "bar_low", "bar_close", "quote_bid_price", "quote_ask_price"])

for _ in range(100):  # Fetch data 100 times (or as needed)
    # Fetch the latest data (reusing your fetch code)
    trade_url = "https://data.alpaca.markets/v2/stocks/trades/latest?symbols=TWTR&feed=iex&currency=USD"
    bar_url = "https://data.alpaca.markets/v2/stocks/bars/latest?symbols=TWTR&feed=iex&currency=USD"
    quote_url = "https://data.alpaca.markets/v2/stocks/TWTR/quotes/latest?feed=iex&currency=USD"

    trade_data = requests.get(trade_url, headers=headers).json()['trades']['TWTR']
    bar_data = requests.get(bar_url, headers=headers).json()['bars']['TWTR']
    quote_data = requests.get(quote_url, headers=headers).json()['quote']

    # Append new data
    new_row = {
        "timestamp": pd.Timestamp.now(),
        "trade_price": trade_data['p'],
        "trade_size": trade_data['s'],
        "bar_open": bar_data['o'],
        "bar_high": bar_data['h'],
        "bar_low": bar_data['l'],
        "bar_close": bar_data['c'],
        "quote_bid_price": quote_data['bp'],
        "quote_ask_price": quote_data['ap'],
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Wait 1 minute (or adjust as needed)
    time.sleep(60)

# Save collected data
df.to_csv("realtime_data.csv", index=False)
print(df)
