import requests

# API Endpoint and Keys
url = "https://data.alpaca.markets/v2/stocks/bars"
headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": "PKCS6JK0RLPEEOCA11KO",
    "APCA-API-SECRET-KEY": "BZ9qKKrhl1wnuGxbUDiL186jPUQGg8SPwW24ikSP",
}

# Query Parameters
params = {
    "symbols": "TWTR",                   # Stock symbol
    "timeframe": "5Min",                 # Timeframe
    "start": "2024-09-03T00:00:00Z",     # Start time (ISO 8601 format)
    "end": "2024-11-03T00:00:00Z",       # End time (ISO 8601 format)
    "limit": 1000,                       # Limit (maximum: 10000 per query)
    "adjustment": "dividend",            # Adjustment for dividends
    "asof": "2024-12-10",                # Fetch data as of a specific date
    "feed": "iex",                       # Data feed (default: `sip` or `iex`)
    "currency": "USD",                   # Currency (default: USD)
    "sort": "asc",                       # Sort order (ascending)
}

# Make the Request
response = requests.get(url, headers=headers, params=params)

# Debug Response
if response.status_code == 200:
    print("Request Successful!")
    print(response.json())  # Print JSON response
else:
    print(f"Error: {response.status_code}")
    print(response.text)  # Print error details
