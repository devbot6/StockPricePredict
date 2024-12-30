import requests
import pandas as pd
from datetime import datetime

# API Keys
API_KEY = "PKCS6JK0RLPEEOCA11KO"
SECRET_KEY = "BZ9qKKrhl1wnuGxbUDiL186jPUQGg8SPwW24ikSP"

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}

# Fetch Latest Trade
trade_url = "https://data.alpaca.markets/v2/stocks/trades/latest?symbols=TWTR&feed=iex&currency=USD"
trade_response = requests.get(trade_url, headers=headers).json()

# Fetch Latest Bar
bar_url = "https://data.alpaca.markets/v2/stocks/bars/latest?symbols=TWTR&feed=iex&currency=USD"
bar_response = requests.get(bar_url, headers=headers).json()

# Fetch Latest Quote
quote_url = "https://data.alpaca.markets/v2/stocks/TWTR/quotes/latest?feed=iex&currency=USD"
quote_response = requests.get(quote_url, headers=headers).json()

# Extract Data
trade_data = trade_response['trades']['TWTR']
bar_data = bar_response['bars']['TWTR']
quote_data = quote_response['quote']

# Create DataFrame
data = {
    "timestamp": [datetime.utcnow()],
    "trade_price": [trade_data['p']],
    "trade_size": [trade_data['s']],
    "bar_open": [bar_data['o']],
    "bar_high": [bar_data['h']],
    "bar_low": [bar_data['l']],
    "bar_close": [bar_data['c']],
    "quote_bid_price": [quote_data['bp']],
    "quote_ask_price": [quote_data['ap']],
}

df = pd.DataFrame(data)

# Save to CSV for further analysis
df.to_csv("stock_data.csv", index=False)
print(df)
