# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement
Stock price prediction is an important task in financial analysis because investors and organizations rely on accurate forecasts to make better investment decisions. Traditional statistical methods often struggle to capture complex patterns in time-series data such as stock prices.
The objective of this project is to develop a Recurrent Neural Network (RNN) model that can learn patterns from historical stock price data and predict future prices. Using the historical closing prices of Google stock, the model will be trained on a training dataset and evaluated on a separate test dataset.
The system will involve loading the datasets, preprocessing the data, building and training an RNN model, and then predicting stock prices for the test dataset. Finally, the predicted values will be compared with the actual stock prices to evaluate the performance and accuracy of the model.

## Theory
A Recurrent Neural Network (RNN) is a type of deep learning model designed to handle sequential data, such as time series like stock prices. It processes previous inputs through loops, allowing it to capture temporal dependencies and patterns over time. When used for stock price prediction, the RNN analyzes historical price data to learn trends and make future price estimates. Its ability to remember information across sequences makes it suitable for modeling the dynamic and seasonal nature of stock markets. Overall, RNNs help improve forecast accuracy by leveraging past data to inform future predictions.

### train.dataset
<img width="980" height="238" alt="Screenshot 2026-03-13 104716" src="https://github.com/user-attachments/assets/89dae5a7-1378-4a22-810a-c539e517d93b" />

### test.dataest
<img width="892" height="242" alt="image" src="https://github.com/user-attachments/assets/8553d5a1-9a7b-4d84-9d93-f91f68146854" />


## DESIGN STEPS
### STEP 1: 
Load and normalize data, create sequences.

### STEP 2: 
Convert data to tensors and set up DataLoader.

### STEP 3: 
Define the RNN model architecture.

### STEP 4: 
Summarize, compile with loss and optimizer.

### STEP 5: 
Train the model with loss tracking.

### STEP 6: 
Predict on test data, plot actual vs. predicted prices.



## PROGRAM

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

## Step 1: Load and Preprocess Data
# Load training and test datasets
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

df_train.head()
df_test.head()

# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

!pip install torchinfo

from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))
  print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')
  print('Name: Sajen Murali')
  print('Register Number: 212223220089')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()
train_model = train_model(model, train_loader, criterion, optimizer, epochs=20)

model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name:Sajen Murali')
print('Register Number: 212223220089')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')




```

### OUTPUT

## Training Loss Over Epochs Plot

![image alt](https://github.com/SajenMurali/DL---RNN/blob/488a2cda33b9f4a675167671159a7bd35ab6ddc7/Screenshot%202026-03-15%20220244.png)

## True Stock Price, Predicted Stock Price vs time and Predictions

![image alt](https://github.com/SajenMurali/DL---RNN/blob/488a2cda33b9f4a675167671159a7bd35ab6ddc7/Screenshot%202026-03-15%20220258.png)


## RESULT
Thi program has been executed successfully.
