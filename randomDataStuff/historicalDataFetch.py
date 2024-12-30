import requests
import pandas as pd

# API Keys
API_KEY = ""
SECRET_KEY = ""

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}

# Fetch Historical Bars
url = "https://data.alpaca.markets/v2/stocks/TWTR/bars"
params = {
    "start": "2023-01-01T00:00:00Z",  # Start date
    "end": "2023-12-01T00:00:00Z",    # End date
    "timeframe": "1Day",              # Timeframe: 1 minute, 1 hour, 1 day
    "feed": "iex",                    # Data feed
    "currency": "USD",
}

response = requests.get(url, headers=headers, params=params)
data = response.json()

# Convert to DataFrame
bars = data.get("bars", {})
df = pd.DataFrame(bars)
print(df)

# Save to CSV
df.to_csv("historical_data.csv", index=False)
