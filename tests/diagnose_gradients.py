import asyncio
import numpy as np
import torch
import torch.nn as nn
from crypto_trading_framework.data_ingestion import DataIngestion
from crypto_trading_framework.core.indicators import add_all_indicators
from crypto_trading_framework.ml.ml_pipeline import MLPipeline
from crypto_trading_framework.ml.model import create_model
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def check_gradients(model, loss_fn, X, y, device):
    model.train()
    x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    output = model(x_tensor)
    loss = loss_fn(output, y_tensor)
    loss.backward()

    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append((name, grad_norm))
        else:
            grad_norms.append((name, 0.0))

    return grad_norms, loss.item()

async def main():
    ingestion = DataIngestion(
        exchange_id=config["data"]["exchange_id"],
        symbol=config["data"]["symbol"]
    )
    try:
        raw_data = await ingestion.fetch_multi_timeframe(
            timeframes=["m15"],
            limit=config["data"]["lookback"]
        )
    except Exception as e:
        print(f"Gagal fetch: {e}")
        raw_data = {}
        df_yf = ingestion.fetch_yfinance(
            ticker=config["data"]["yfinance_ticker"],
            period=config["data"]["yfinance_period"],
            interval="15m"
        )
        if df_yf is not None:
            raw_data["m15"] = df_yf
    await ingestion.close()

    if "m15" not in raw_data:
        print("Tidak ada data m15")
        return

    df = add_all_indicators(raw_data["m15"])
    pipeline = MLPipeline()
    df = pipeline.define_target(df, forward_periods=config["ml"]["forward_periods"])
    df = df.drop_nulls()

    if df.is_empty():
        print("Data kosong")
        return

    features, feature_cols = pipeline.prepare_features(df, feature_cols=config["ml"]["feature_cols"])
    targets = df.select("target").to_numpy().flatten()
    scaled = pipeline.scale_features(features, fit=True)
    X, y = pipeline.create_sequences(scaled, targets, time_steps=config["ml"]["time_steps"])

    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_type="lstm",
        input_size=X.shape[2],
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    print("\n=== Gradient Check (Initial) ===")
    grads, loss = check_gradients(model, loss_fn, X[:100], y[:100], device)
    for name, g in grads:
        print(f"  {name:40s}: {g:.6f}")

    print(f"\nInitial loss: {loss:.4f}")

    for epoch in range(5):
        grads, loss = check_gradients(model, loss_fn, X[:100], y[:100], device)
        print(f"\nEpoch {epoch+1} loss: {loss:.4f}")
        avg_grad = np.mean([g for _, g in grads])
        print(f"  Average grad norm: {avg_grad:.6f}")

if __name__ == "__main__":
    asyncio.run(main())
