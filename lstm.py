import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the data
df = pd.read_csv('cleaned_HistoricalData_file.csv')

# Inspect the first few rows of the data
print(df.head())

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')


# Set timestamp as index (if needed)
df.set_index('timestamp', inplace=True)

# Select relevant features (you can choose which features you want to use)
features = ['trade_price', 'trade_size']

# You can optionally create your target feature (for example, future price)
# For simplicity, let's predict the 'bar_close' as the target
target = 'trade_price'

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features, index=df.index)

# Optionally, for target scaling (if needed):
target_scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled[target] = target_scaler.fit_transform(df[[target]])

# Create sequences for LSTM (using a sliding window approach)
def create_sequences(data, target_column, seq_length):
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length][features].values
        target = data.iloc[i+seq_length][target_column]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Set the sequence length (e.g., 60 days of data for each prediction)
seq_length = 60

# Create sequences
X, y = create_sequences(df_scaled, target, seq_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader for training and testing
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Passing through LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Only get the output of the last time step
        out = self.fc(lstm_out[:, -1, :])
        
        return out

# Hyperparameters
input_size = len(features)  # Number of input features
hidden_size = 64  # Number of LSTM units
num_layers = 2  # Number of LSTM layers
output_size = 1  # Predicting one value (stock price)

# Instantiate the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)


# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error (MSE) for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 80  # Number of epochs

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients
        
        # Forward pass
        y_pred = model(X_batch)
        
        # Compute the loss
        loss = criterion(y_pred, y_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    # Print loss for every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()  # Set model to evaluation mode
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

# Rescale the predictions and actual values
test_preds_rescaled = target_scaler.inverse_transform(test_preds)
test_labels_rescaled = target_scaler.inverse_transform(test_labels)

# Plot predictions vs actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(test_labels_rescaled, label='True Prices')
plt.plot(test_preds_rescaled, label='Predicted Prices')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'stock_lstm_model.pth')
