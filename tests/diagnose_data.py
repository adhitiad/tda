import asyncio
import numpy as np
import polars as pl
from crypto_trading_framework.data_ingestion import DataIngestion
from crypto_trading_framework.core.indicators import add_all_indicators
from crypto_trading_framework.ml.ml_pipeline import MLPipeline
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

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
    print(f"After indicators: {len(df)} rows")
    print(f"Columns: {df.columns}")

    pipeline = MLPipeline()
    df = pipeline.define_target(df, forward_periods=config["ml"]["forward_periods"])
    df = df.drop_nulls()

    print(f"\nAfter drop_nulls: {len(df)} rows")

    if df.is_empty():
        print("Data kosong")
        return

    features, feature_cols = pipeline.prepare_features(df, feature_cols=config["ml"]["feature_cols"])
    print(f"\nFeatures shape: {features.shape}")
    print(f"Feature cols: {feature_cols}")

    print("\nFeature statistics (before scaling):")
    for i, col in enumerate(feature_cols):
        col_data = features[:, i]
        print(f"  {col:30s}: min={col_data.min():.4f}, max={col_data.max():.4f}, mean={col_data.mean():.4f}, std={col_data.std():.4f}")

    scaled = pipeline.scale_features(features, fit=True)
    print("\nFeature statistics (after scaling):")
    for i, col in enumerate(feature_cols):
        col_data = scaled[:, i]
        print(f"  {col:30s}: min={col_data.min():.4f}, max={col_data.max():.4f}, mean={col_data.mean():.4f}, std={col_data.std():.4f}")

    targets = df.select("target").to_numpy().flatten()
    print(f"\nTarget distribution: {np.unique(targets, return_counts=True)}")

    X, y = pipeline.create_sequences(scaled, targets, time_steps=config["ml"]["time_steps"])
    print(f"\nSequences shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"X min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}")

if __name__ == "__main__":
    asyncio.run(main())
