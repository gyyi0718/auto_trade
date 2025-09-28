import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas_ta as ta
import joblib


class PriceDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TransformerPricePredictor(nn.Module):
    def __init__(self, seq_len, input_dim=6, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[-1]
        x = self.output_proj(x)
        return x


def train_model(model, train_loader, val_loader=None, epochs=50, lr=1e-3):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    pred_val = model(x_val)
                    val_loss += criterion(pred_val, y_val).item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            model.train()
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}")

    return train_losses, val_losses


def rolling_prediction(model, data, seq_len, scaler):
    model.eval()
    predictions = []
    actuals = []

    for i in range(len(data) - seq_len):
        x_seq = torch.tensor(data[i:i + seq_len].reshape(1, seq_len, data.shape[1]), dtype=torch.float32)
        with torch.no_grad():
            pred = model(x_seq).numpy().flatten()
        predictions.append(pred)
        actuals.append(data[i + seq_len])

    predictions = scaler.inverse_transform(np.array(predictions))
    actuals = scaler.inverse_transform(np.array(actuals))

    pred_close = predictions[:, 3]
    actual_close = actuals[:, 3]

    rmse = mean_squared_error(actual_close, pred_close, squared=False)
    mae = mean_absolute_error(actual_close, pred_close)
    r2 = r2_score(actual_close, pred_close)

    print(f"\n✅ Evaluation Metrics (Close only):")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.4f}")

    plt.plot(actual_close, label='Actual Close')
    plt.plot(pred_close, label='Predicted Close')
    plt.legend()
    plt.title("Transformer Rolling Prediction - Close")
    plt.show()


if __name__ == "__main__":
    train_df = pd.read_csv("1000PEPEUSDC_train_250727.csv")
    val_df = pd.read_csv("1000PEPEUSDC_val_250727.csv")

    for df in [train_df, val_df]:
        df.ta.ema(length=10, append=True)
        df.ta.rsi(length=14, append=True)
        df.dropna(inplace=True)

    train_price = train_df[['High', 'Low', 'Open', 'Close', 'EMA_10', 'RSI_14']].values
    val_price = val_df[['High', 'Low', 'Open', 'Close', 'EMA_10', 'RSI_14']].values

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_price)
    val_scaled = scaler.transform(val_price)

    joblib.dump(scaler, "1000PEPEUSDC_scaler_250727.pkl")

    SEQ_LEN = 60
    train_dataset = PriceDataset(train_scaled, seq_len=SEQ_LEN)
    val_dataset = PriceDataset(val_scaled, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = TransformerPricePredictor(seq_len=SEQ_LEN, input_dim=train_scaled.shape[1])
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, lr=1e-4)

    torch.save(model.state_dict(), "transformer_price_model.pth")

    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    print("\n📈 Validation Performance")
    rolling_prediction(model, val_scaled, seq_len=SEQ_LEN, scaler=scaler)
