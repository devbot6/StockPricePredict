# Stock Price Prediction Using LSTM

This project uses historical stock data to predict future stock prices using an LSTM (Long Short-Term Memory) model built with PyTorch. The data is fetched via the Alpaca API, processed and cleaned using pandas, and visualized using Matplotlib. A Flask web application was developed to allow users to input data and receive predictions.

## Project Overview

The model predicts the future stock price of a given stock symbol (e.g., TSLA) using historical data, including trades, quotes, and bars. The data is preprocessed and normalized, followed by sequence creation for training the LSTM model. After training, the model is evaluated, and the results are displayed through various visualizations.

Key features:
- Fetches historical stock data using Alpaca API.
- Data cleaning and preprocessing using pandas.
- LSTM model built and trained in PyTorch for stock price prediction.
- Flask web application for real-time predictions.

## Requirements

- Python 3.x
- Libraries: `requests`, `pandas`, `numpy`, `scikit-learn`, `torch`, `matplotlib`, `Flask`

