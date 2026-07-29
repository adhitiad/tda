import asyncio
import numpy as np
import polars as pl
from crypto_trading_framework.data_ingestion import DataIngestion
from crypto_trading_framework.core.indicators import add_all_indicators
from crypto_trading_framework.ml.ml_pipeline import MLPipeline

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

    print("Feature correlations with target:")
    for i, col in enumerate(cols):
        corr = np.corrcoef(features[:, i], targets)[0, 1]
        print(f"  {col}: {corr:.4f}")

    print(f"\nTarget std: {targets.std():.4f}")
    print(f"Random baseline accuracy: {max(targets.mean(), 1 - targets.mean()):.2%}")

if __name__ == "__main__":
    asyncio.run(diagnose())
