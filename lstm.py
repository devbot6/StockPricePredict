import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Load the data
df = pd.read_csv('cleaned_HistoricalData_file2.csv')

# Print available columns to verify
print("Available columns:", df.columns.tolist())

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
df.set_index('timestamp', inplace=True)

# Get all numerical features automatically
features = ['bar_open', 'bar_high', 'bar_low', 'bar_close', 'volume', 
                           'trade_count', 'vwap', 'price_change', 'volume_change', 'high_low_spread', 
                           'open_close_spread', 'ma5', 'ma20', 'volatility']
print("\nNumerical features found:", features)

target = 'ma5'

print(features)

# Remove target from input features to avoid data leakage
input_features = [f for f in features]
print(f"\nTarget variable: {target}")
print(f"Input features: {input_features}")

# Normalize all features
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features, index=df.index)

# Create sequences for LSTM
def create_sequences(data, target_column, input_columns, seq_length):
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length][input_columns].values
        target = data.iloc[i+seq_length][target_column]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Set sequence length
seq_length = 60

# Create sequences
X, y = create_sequences(df_scaled, target, input_features, seq_length)

print(f"\nSequence shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Modified hyperparameters
input_size = len(input_features)
hidden_size = 128
num_layers = 2
output_size = 1

# Initialize model, criterion, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 80
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_preds = []
    test_labels = []
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        test_preds.append(y_pred.numpy())
        test_labels.append(y_batch.numpy())

# Convert predictions and labels to numpy arrays
test_preds = np.concatenate(test_preds, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

# Create a DataFrame for inverse transform
pred_df = pd.DataFrame(np.zeros((len(test_preds), len(features))), columns=features)
pred_df[target] = test_preds
label_df = pd.DataFrame(np.zeros((len(test_labels), len(features))), columns=features)
label_df[target] = test_labels

# Rescale predictions and actual values
test_preds_rescaled = scaler.inverse_transform(pred_df)[range(len(test_preds)), features.index(target)]
test_labels_rescaled = scaler.inverse_transform(label_df)[range(len(test_labels)), features.index(target)]

# Calculate accuracy metrics
def calculate_metrics(y_true, y_pred):
    # R² Score
    r2 = r2_score(y_true, y_pred)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Direction Accuracy (percentage of correct direction predictions)
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    direction_accuracy = np.mean(direction_true == direction_pred) * 100
    
    return r2, rmse, mape, direction_accuracy

# Calculate and print all metrics
r2, rmse, mape, direction_accuracy = calculate_metrics(test_labels_rescaled, test_preds_rescaled)

print("\nModel Performance Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
print(f"Direction Accuracy: {direction_accuracy:.2f}%")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(test_labels_rescaled, label='True Prices')
plt.plot(test_preds_rescaled, label='Predicted Prices')
plt.title(f'Price Prediction (R² = {r2:.4f}, Direction Accuracy = {direction_accuracy:.2f}%)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Define the save path
save_path = 'stock_lstm_model2.pth'

# Try to save the model and verify it was saved
try:
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'features': features,
        'input_features': input_features,
        'target': target,
        'scaler': scaler,
        'metrics': {
            'r2': r2,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }
    }, save_path)
    
    # Verify the file was created
    if os.path.exists(save_path):
        print(f"\nModel successfully saved to: {os.path.abspath(save_path)}")
        print(f"File size: {os.path.getsize(save_path) / 1024:.2f} KB")
    else:
        print("\nError: Model file was not created")
        
except Exception as e:
    print(f"\nError saving model: {str(e)}")
    print("Current working directory:", os.getcwd())
