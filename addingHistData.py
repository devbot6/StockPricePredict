import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

# API Keys
API_KEY = ""
SECRET_KEY = ""

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}

def get_historical_data(symbol, lookback_days=30):
    # Calculate dates in EST timezone
    est = pytz.timezone('US/Eastern')
    end_date = datetime.now(est)
    start_date = end_date - timedelta(days=lookback_days)
    
    # Format dates for API
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    print(f"Fetching data from {start_str} to {end_str}")
    
    # Parameters for data fetching
    params = {
        "symbols": symbol,
        "start": start_str,
        "end": end_str,
        "limit": 10000,  # Increased limit for more data
        "timeframe": "1Hour"
    }
    
    try:
        # Fetch Historical Bars (more reliable than individual trades)
        bar_url = "https://data.alpaca.markets/v2/stocks/bars"
        bar_response = requests.get(bar_url, headers=headers, params={**params, "adjustment": "all"})
        bar_response.raise_for_status()  # Raise exception for bad status codes
        bar_data = bar_response.json()['bars'][symbol]
        
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
                    "trade_count": bar["n"],
                    "vwap": bar["vw"]
                }
                for bar in bar_data
            ]
        )
        
        # Convert timestamp to datetime
        bar_df['timestamp'] = pd.to_datetime(bar_df['timestamp'])
        
        # Sort by timestamp
        bar_df = bar_df.sort_values('timestamp')
        
        # Add some derived features
        bar_df['price_change'] = bar_df['bar_close'].pct_change()
        bar_df['volume_change'] = bar_df['volume'].pct_change()
        bar_df['high_low_spread'] = bar_df['bar_high'] - bar_df['bar_low']
        bar_df['open_close_spread'] = bar_df['bar_close'] - bar_df['bar_open']
        
        # Calculate moving averages
        bar_df['ma5'] = bar_df['bar_close'].rolling(window=5).mean()
        bar_df['ma20'] = bar_df['bar_close'].rolling(window=20).mean()
        
        # Calculate volatility
        bar_df['volatility'] = bar_df['price_change'].rolling(window=5).std()
        
        # Save to CSV
        file_name = f"{symbol}_historical_data.csv"
        bar_df.to_csv(file_name, index=False)
        print(f"Data saved to {file_name}")
        print(f"Shape of data: {bar_df.shape}")
        print("\nFirst few rows:")
        print(bar_df.head())
        print("\nMissing values:")
        print(bar_df.isnull().sum())
        
        return bar_df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except KeyError as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    # Get data for Tesla stock
    symbol = 'TSLA'
    df = get_historical_data(symbol, lookback_days=30)  # Get last 30 days of data
    
    if df is not None:
        # Display some basic statistics
        print("\nBasic statistics:")
        print(df.describe())