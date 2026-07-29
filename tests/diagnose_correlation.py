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
    pipeline = MLPipeline()
    df = pipeline.define_target(df, forward_periods=config["ml"]["forward_periods"])
    df = df.drop_nulls()

    if df.is_empty():
        print("Data kosong")
        return

    feature_cols = [c for c in config["ml"]["feature_cols"] if c in df.columns]
    print(f"Feature columns available: {feature_cols}")
    print(f"Total samples: {len(df)}")

    print("\nKorelasi dengan target:")
    correlations = {}
    for col in feature_cols:
        try:
            corr = df.select(pl.corr(col, "target")).item()
            correlations[col] = corr
            print(f"  {col:30s}: {corr:+.4f}")
        except Exception as e:
            print(f"  {col:30s}: ERROR - {e}")

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\nTop 10 fitur paling prediktif:")
    for col, corr in sorted_corr[:10]:
        print(f"  {col:30s}: {corr:+.4f}")

if __name__ == "__main__":
    asyncio.run(main())
