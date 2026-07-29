import asyncio
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

    ob = await ingestion.fetch_orderbook_async(limit=100)
    if ob is not None:
        print(f"Orderbook: {ob.to_dicts()}")

    await ingestion.close()

    if "m15" not in raw_data:
        print("Tidak ada data m15")
        return

    df = add_all_indicators(raw_data["m15"])
    print(f"\nADX stats:")
    adx_stats = df.select(
        pl.col("adx").min().alias("min"),
        pl.col("adx").max().alias("max"),
        pl.col("adx").mean().alias("mean"),
        pl.col("adx").std().alias("std"),
    ).to_dicts()[0]
    print(f"  min={adx_stats['min']:.2f}, max={adx_stats['max']:.2f}, mean={adx_stats['mean']:.2f}, std={adx_stats['std']:.2f}")

    print(f"\nRegime distribution:")
    regime_dist = df.group_by("regime_trending").agg(pl.len().alias("count")).to_dicts()
    for row in regime_dist:
        print(f"  Regime {row['regime_trending']}: {row['count']} bars")

if __name__ == "__main__":
    asyncio.run(main())
