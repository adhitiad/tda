import asyncio
import numpy as np
import polars as pl
from crypto_trading_framework.data_ingestion import DataIngestion
from crypto_trading_framework.core.indicators import add_all_indicators
from crypto_trading_framework.ml.ml_pipeline import MLPipeline
from crypto_trading_framework.ml.model import create_model
import torch
import torch.nn as nn

async def diagnose():
    ingestion = DataIngestion(exchange_id="tokocrypto", symbol="BTC/USDT")
    raw_data = await ingestion.fetch_multi_timeframe(timeframes=["h4"], limit=500)
    await ingestion.close()

    df = add_all_indicators(raw_data["h4"])
    pipeline = MLPipeline()
    df = pipeline.define_target(df, forward_periods=10)
    df = df.drop_nulls()

    features, cols = pipeline.prepare_features(df)
    targets = df.select("target").to_numpy().flatten()

    print(f"Target distribution: {np.bincount(targets.astype(int))}")
    print(f"Class 1 ratio: {targets.mean():.2%}")

    scaled = pipeline.scale_features(features, fit=True)
    X, y = pipeline.create_sequences(scaled, targets, time_steps=30)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model("lstm", input_size=X_train.shape[2], hidden_size=128, num_layers=2, dropout=0.3).to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    print("\nTraining for 100 epochs...")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(x_test_tensor)
        preds = torch.sigmoid(logits).cpu().numpy().flatten()

    print(f"\nAfter training:")
    print(f"  Min: {preds.min():.4f}")
    print(f"  Max: {preds.max():.4f}")
    print(f"  Mean: {preds.mean():.4f}")
    print(f"  Std: {preds.std():.4f}")
    print(f"  >0.5: {(preds > 0.5).sum()}/{len(preds)}")
    print(f"  >0.75: {(preds > 0.75).sum()}/{len(preds)}")
    print(f"  >0.9: {(preds > 0.9).sum()}/{len(preds)}")

if __name__ == "__main__":
    asyncio.run(diagnose())
