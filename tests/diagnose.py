import asyncio
import numpy as np
import polars as pl
from crypto_trading_framework.data_ingestion import DataIngestion
from crypto_trading_framework.core.indicators import add_all_indicators
from crypto_trading_framework.ml.ml_pipeline import MLPipeline

async def diagnose():
    ingestion = DataIngestion(exchange_id="tokocrypto", symbol="BTC/USDT")
    raw_data = await ingestion.fetch_multi_timeframe(timeframes=["h4"], limit=200)
    await ingestion.close()

    df = add_all_indicators(raw_data["h4"])
    pipeline = MLPipeline()
    df = pipeline.define_target(df, forward_periods=5)
    df = df.drop_nulls()

    features, cols = pipeline.prepare_features(df)
    targets = df.select("target").to_numpy().flatten()

    print(f"Data shape: {features.shape}")
    print(f"Target distribution: {np.bincount(targets.astype(int))}")
    print(f"Class 1 ratio: {targets.mean():.2%}")

    scaled = pipeline.scale_features(features, fit=True)
    X, y = pipeline.create_sequences(scaled, targets, time_steps=20)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train target distribution: {np.bincount(y_train.astype(int))}")
    print(f"Test target distribution: {np.bincount(y_test.astype(int))}")

    from crypto_trading_framework.ml.model import create_model
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model("lstm", input_size=X_train.shape[2]).to(device)

    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = model(x_tensor).cpu().numpy().flatten()

    print(f"\nPrediction distribution:")
    print(f"  Min: {preds.min():.4f}")
    print(f"  Max: {preds.max():.4f}")
    print(f"  Mean: {preds.mean():.4f}")
    print(f"  Std: {preds.std():.4f}")
    print(f"  >0.5: {(preds > 0.5).sum()}/{len(preds)} ({(preds > 0.5).mean():.2%})")
    print(f"  >0.75: {(preds > 0.75).sum()}/{len(preds)} ({(preds > 0.75).mean():.2%})")
    print(f"  >0.9: {(preds > 0.9).sum()}/{len(preds)} ({(preds > 0.9).mean():.2%})")

    if len(preds) > 0:
        print(f"\nTop 5 predictions: {np.sort(preds)[-5:]}")

if __name__ == "__main__":
    asyncio.run(diagnose())
