import torch
import numpy as np
from flask import Flask, request, jsonify
from model import LSTMModel  # Assuming you have your model architecture in a file 'model.py'

# Load the trained model
model = LSTMModel(input_size=2, hidden_size=64, num_layers=2, output_size=1)  # Specify your model params
model.load_state_dict(torch.load("stock_lstm_model.pth"))
model.eval()

# Initialize Flask app
app = Flask(__name__)

# Prediction function
def predict(input_data):
    # Preprocess the input data as required by your model (e.g., normalization)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    return output.numpy().tolist()

# Define API endpoint
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Get data from request
        data = request.json
        input_data = data.get('input')
        if input_data is None:
            return jsonify({"error": "Input data is missing"}), 400
        
        # Get prediction
        prediction = predict(input_data)
        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
