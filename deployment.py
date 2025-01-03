from flask import Flask, render_template, jsonify, render_template_string
import torch
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import datetime

app = Flask(__name__)

# Load the saved model
def load_model():
    try:
        checkpoint = torch.load('stock_lstm_model2.pth')
        
        # Recreate model architecture
        class LSTMModel(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size=1):
                super(LSTMModel, self).__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = torch.nn.Linear(hidden_size, output_size)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return out

        # Initialize model with same parameters
        model = LSTMModel(
            input_size=len(checkpoint['input_features']),
            hidden_size=128,
            num_layers=2,
            output_size=1
        )
        
        # Load saved parameters
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, checkpoint
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def calculate_indicators(df):
    """Calculate technical indicators matching the training features"""
    # Map yfinance data to our training features
    df['bar_open'] = df['Open']
    df['bar_high'] = df['High']
    df['bar_low'] = df['Low']
    df['bar_close'] = df['Close']
    df['volume'] = df['Volume']
    
    # Calculate additional features
    df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['volume_change'] = df['Volume'].pct_change()
    df['high_low_spread'] = (df['High'] - df['Low']) / df['Low'] * 100
    df['open_close_spread'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma20'] = df['Close'].rolling(window=20).mean()  # 20-day moving average
    df['trade_count'] = df['Volume'].rolling(window=5).mean()  # Approximation since yfinance doesn't provide trade count
    
    # New features
    df['price_change'] = df['Close'].pct_change() * 100  # Percentage price change
    df['volatility'] = (df['High'] - df['Low']) / df['Close'] * 100  # Intraday volatility

    # Drop rows with NaN values from rolling calculations
    df = df.dropna()
    
    return df


# Get latest stock data
def get_stock_data():
    try:
        # Get Tesla stock data for the last 60 days
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=70)  # Get extra days for safety
        tsla = yf.download('TSLA', start=start_date, end=end_date, interval='1d')
        
        # Calculate all required features
        tsla = calculate_indicators(tsla)
        
        # Drop any NaN values
        tsla = tsla.dropna()
        
        return tsla
    except Exception as e:
        print(f"Error getting stock data: {str(e)}")
        return None

# Make prediction
def predict_stock_movement():
    model, checkpoint = load_model()
    if model is None or checkpoint is None:
        return {"error": "Failed to load model"}
    
    # Get latest data
    stock_data = get_stock_data()
    if stock_data is None:
        return {"error": "Failed to get stock data"}
    
    try:
        # Print available features for debugging
        print("Available features in stock_data:", stock_data.columns.tolist())
        print("Required features from model:", checkpoint['features'])
        
        # Prepare data with required features
        required_features = ['bar_open', 'bar_high', 'bar_low', 'bar_close', 'volume', 
                           'trade_count', 'vwap', 'price_change', 'volume_change', 'high_low_spread', 
                           'open_close_spread','ma5', 'ma20', 'volatility']
        
        feature_data = stock_data[required_features].copy()
        
        # Scale the data
        scaler = checkpoint['scaler']
        scaled_data = scaler.transform(feature_data.tail(60))
        
        # Convert to tensor
        X = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(X)
        
        # Convert prediction back to price
        pred_df = pd.DataFrame(np.zeros((1, len(required_features))), columns=required_features)
        pred_df[checkpoint['target']] = prediction.numpy()
        pred_price = scaler.inverse_transform(pred_df)[0, required_features.index(checkpoint['target'])]
        
        # Get current price
        current_price = stock_data['Close'].iloc[-1]
        
        # Determine if price will go up or down
        prediction_direction = "UP 📈" if pred_price > current_price else "DOWN 📉"
        confidence = abs((pred_price - current_price) / current_price * 100)
        
        return {
            "direction": prediction_direction,
            "confidence": f"{confidence:.2f}%",
            "current_price": f"${current_price:.2f}",
            "predicted_price": f"${pred_price:.2f}"
        }
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return {"error": f"Prediction error: {str(e)}"}

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tesla Stock Predictor</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/axios/1.6.2/axios.min.js"></script>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .predict-btn {
            display: block;
            width: 200px;
            margin: 20px auto;
            padding: 15px 30px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .predict-btn:hover {
            background-color: #0056b3;
        }
        .predict-btn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
            font-size: 18px;
            display: none;
        }
        .up {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .down {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .error {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .details {
            margin-top: 20px;
            font-size: 16px;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Tesla Stock Predictor</h1>
        <button id="predictBtn" class="predict-btn">Predict Stock Movement</button>
        <div id="loading" class="loading">Analyzing market data...</div>
        <div id="result" class="result"></div>
    </div>

    <script>
        const predictBtn = document.getElementById('predictBtn');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');

        predictBtn.addEventListener('click', async () => {
            try {
                // Disable button and show loading
                predictBtn.disabled = true;
                loading.style.display = 'block';
                result.style.display = 'none';

                // Make prediction request
                const response = await axios.get('/predict');
                const data = response.data;

                // Create result message
                let resultHTML = '';
                if (data.error) {
                    result.className = 'result error';
                    resultHTML = `<strong>Error:</strong> ${data.error}`;
                } else {
                    result.className = `result ${data.direction.includes('UP') ? 'up' : 'down'}`;
                    resultHTML = `
                        <strong>Prediction: Stock will go ${data.direction}</strong>
                        <div class="details">
                            <p>Current Price: ${data.current_price}</p>
                            <p>Predicted Price: ${data.predicted_price}</p>
                            <p>Confidence: ${data.confidence}</p>
                        </div>
                    `;
                }

                // Show result
                result.innerHTML = resultHTML;
                result.style.display = 'block';
            } catch (error) {
                result.className = 'result error';
                result.innerHTML = `<strong>Error:</strong> Failed to get prediction`;
                result.style.display = 'block';
            } finally {
                // Re-enable button and hide loading
                predictBtn.disabled = false;
                loading.style.display = 'none';
            }
        });
    </script>
</body>
</html>
    ''')

@app.route('/predict')
def predict():
    result = predict_stock_movement()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)