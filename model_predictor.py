import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "gru_model.pth")

def predict_stock(stock_name, stock_data, return_recommendation=False):
    print(f"\n---------------------------------------For Stock {stock_name}---------------------------------------")

    if stock_name == 'JSW':
        sentiment_column = 'JSW Steel_avg_sentiment'
    else:
        sentiment_column = f"{stock_name.replace(' ', '')}_avg_sentiment"
    if sentiment_column not in stock_data.columns:
        raise ValueError(f"Sentiment column '{sentiment_column}' not found in the data.")
    
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values('date')
    stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    
    features = stock_data[['High', 'Low', sentiment_column, 'Open', 'Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    def create_sequences(data, seq_length):
        xs = []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            xs.append(x)
        return torch.tensor(xs, dtype=torch.float32)

    seq_length = 90
    X = create_sequences(scaled_features, seq_length)

    class GRUModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super(GRUModel, self).__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 2)  

        def forward(self, x):
            h0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            return out

    input_dim = 5
    hidden_dim = 512
    num_layers = 2
    model = GRUModel(input_dim, hidden_dim, num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    last_sequence = torch.tensor(scaled_features[-seq_length:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        next_day_prediction = model(last_sequence).numpy()
        next_day_prediction = scaler.inverse_transform([[next_day_prediction[0, 0], next_day_prediction[0, 1], 0, 0, 0]])[0]

    predicted_high = next_day_prediction[0]
    predicted_low = next_day_prediction[1]

    print(f"Predicted High for the Next Day: {predicted_high:.2f}")
    print(f"Predicted Low for the Next Day: {predicted_low:.2f}")

    if not return_recommendation:
        return predicted_high, predicted_low

    latest_sma_10 = stock_data['SMA_10'].iloc[-1]
    latest_sma_50 = stock_data['SMA_50'].iloc[-1]
    latest_close = stock_data['Close'].iloc[-1]

    if latest_sma_10 > latest_sma_50 and latest_close > latest_sma_10:
        recommendation = "Buy"
    elif latest_sma_10 < latest_sma_50 and latest_close < latest_sma_10:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    print(f"Recommendation for {stock_name}: {recommendation}")

    return recommendation

