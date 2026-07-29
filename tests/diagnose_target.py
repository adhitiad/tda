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
        symbol=config["data"]["symbols"][0]
    )
    try:
        raw_data = await ingestion.fetch_multi_timeframe(
            timeframes=config["data"]["timeframes"],
            limit=config["data"]["lookback"]
        )
    except Exception as e:
        print(f"Gagal fetch: {e}")
        raw_data = {}
        for tf in config["data"]["timeframes"]:
            interval_map = {"m5": "5m", "m15": "15m", "m30": "30m", "h1": "1h", "h4": "4h", "d1": "1d"}
            df_yf = ingestion.fetch_yfinance(
                ticker=config["data"]["yfinance_ticker"][0],
                period=config["data"]["yfinance_period"],
                interval=interval_map.get(tf, "1h")
            )
            if df_yf is not None:
                raw_data[tf] = df_yf
    await ingestion.close()

    for tf, df_raw in raw_data.items():
        print(f"\n=== {tf} ===")
        df = add_all_indicators(df_raw)
        pipeline = MLPipeline()
        df = pipeline.define_target(df, forward_periods=config["ml"]["forward_periods"])
        df = df.drop_nulls()

        if df.is_empty():
            print("Data kosong setelah preprocessing")
            continue

        targets = df.select("target").to_numpy().flatten()
        unique, counts = np.unique(targets, return_counts=True)
        print(f"Total samples: {len(targets)}")
        for u, c in zip(unique, counts):
            print(f"  Class {int(u)}: {c} ({c/len(targets)*100:.2f}%)")

if __name__ == "__main__":
    asyncio.run(main())
