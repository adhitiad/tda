"""
Data reconciliation utility for TimescaleDB + CCXT.

Detects missing OHLCV candles by comparing database timestamps against
exchange data, interpolates absent candles via linear interpolation,
and persists them with `is_interpolated = 1.0`. Feature store rows in
the affected range are also marked accordingly.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import polars as pl
import yaml

from crypto_trading_framework.core.config_schema import validate_config
from crypto_trading_framework.core.logging import get_logger
from crypto_trading_framework.data_ingestion import DataIngestion
from crypto_trading_framework.db.database import (
    query_ohlcv,
    upsert_ohlcv_dataframe,
)
from crypto_trading_framework.ml.feature_store import (
    compute_and_store_features,
)

logger = get_logger("reconciler")


class DataReconciler:
    """Compares TimescaleDB candles against CCXT exchange data and fills gaps."""

    TIMEFRAME_MS: ClassVar[dict[str, int]] = {
        "m15": 15 * 60 * 1000,
        "m30": 30 * 60 * 1000,
        "h1": 60 * 60 * 1000,
        "h4": 4 * 60 * 60 * 1000,
        "d1": 24 * 60 * 60 * 1000,
    }

    TIMEFRAME_INTERVAL: ClassVar[dict[str, str]] = {
        "m15": "15m",
        "m30": "30m",
        "h1": "1h",
        "h4": "4h",
        "d1": "1d",
    }

    def __init__(self, config: dict[str, Any]):
        self.exchange_id = config["data"]["exchange_id"]
        self.symbols = config["data"]["symbols"]
        self.timeframes = config["data"]["timeframes"]
        self.lookback = int(config["data"].get("lookback", 2000))

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "-").replace(":", "")

    async def reconcile(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "gaps_found": 0,
            "candles_interpolated": 0,
            "candles_inserted": 0,
        }

        normalized_symbol = self._normalize_symbol(symbol)

        db_pdf = query_ohlcv(
            symbol=normalized_symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )

        with DataIngestion(self.exchange_id, symbol) as ingestion:
            exchange_df = await ingestion.fetch_ohlcv_async(
                timeframe=timeframe,
                limit=max(self.lookback, 1000),
            )

        if exchange_df is None or exchange_df.is_empty():
            logger.warning(f"[Reconciler] No exchange data returned for {symbol} {timeframe}")
            return summary

        expected_ts = exchange_df["timestamp"]

        if not db_pdf.empty:
            db_pdf = db_pdf.copy()
            db_pdf["timestamp"] = pd.to_datetime(db_pdf["timestamp"])
        db_polars_load = pl.from_pandas(db_pdf)
        if not db_polars_load.is_empty():
            existing_ts = db_polars_load["timestamp"]
        else:
            existing_ts = pl.Series([], dtype=pl.Datetime("us"))

        existing_set = set(existing_ts.to_list())
        gap_timestamps = expected_ts.filter(~expected_ts.is_in(existing_set))

        if gap_timestamps.is_empty():
            logger.info(f"[Reconciler] No gaps for {symbol} {timeframe}")
            return summary

        summary["gaps_found"] = int(gap_timestamps.len())

        interpolated_df = self._interpolate_gaps(exchange_df, gap_timestamps, timeframe)
        if interpolated_df.is_empty():
            return summary

        interpolated_df = interpolated_df.with_columns(
            pl.lit(1.0).alias("is_interpolated")
        )

        pdf = interpolated_df.to_pandas()
        pdf["symbol"] = normalized_symbol
        pdf["timeframe"] = timeframe
        pdf["source"] = "ccxt"
        pdf["updated_at"] = pd.Timestamp.now("UTC")

        rows = upsert_ohlcv_dataframe(pdf, source="ccxt")
        summary["candles_interpolated"] = int(interpolated_df.height)
        summary["candles_inserted"] = int(rows)

        logger.info(
            f"[Reconciler] {symbol} {timeframe}: gaps={gap_timestamps.len()}, "
            f"interpolated={interpolated_df.height}, inserted={rows}"
        )

        self._recompute_features(normalized_symbol, timeframe, gap_timestamps)

        return summary

    def _interpolate_gaps(
        self,
        exchange_df: pl.DataFrame,
        gap_timestamps: pl.Series,
        timeframe: str,
    ) -> pl.DataFrame:
        if gap_timestamps.is_empty():
            return pl.DataFrame()

        interval = self.TIMEFRAME_INTERVAL.get(timeframe, timeframe)
        gap_start: datetime = gap_timestamps.min()  # type: ignore[assignment]
        gap_end: datetime = gap_timestamps.max()  # type: ignore[assignment]
        full_range = pl.datetime_range(
            start=gap_start,
            end=gap_end,
            interval=interval,
            eager=True,
        )
        full_df = pl.DataFrame({"timestamp": full_range})

        df_sorted = exchange_df.sort("timestamp")
        numeric_cols = [c for c in df_sorted.columns if c != "timestamp"]

        merged = full_df.join(df_sorted, on="timestamp", how="left")
        for col in numeric_cols:
            if col not in merged.columns:
                continue
            merged = merged.with_columns(
                pl.col(col).interpolate().alias(col)
            )

        result = merged.filter(
            pl.col("timestamp").is_in(gap_timestamps.to_list())
        )

        if "open" in result.columns and "high" in result.columns and "low" in result.columns and "close" in result.columns:
            result = result.with_columns([
                pl.when(pl.col("high") < pl.max_horizontal(pl.col("open"), pl.col("close")))
                .then(pl.max_horizontal(pl.col("open"), pl.col("close")))
                .otherwise(pl.col("high"))
                .alias("high"),
                pl.when(pl.col("low") > pl.min_horizontal(pl.col("open"), pl.col("close")))
                .then(pl.min_horizontal(pl.col("open"), pl.col("close")))
                .otherwise(pl.col("low"))
                .alias("low"),
            ])

        return result

    def _recompute_features(
        self,
        symbol: str,
        timeframe: str,
        gap_timestamps: pl.Series,
    ) -> None:
        buffer_candles = 200
        tf_ms = self.TIMEFRAME_MS.get(timeframe, 3600000)

        gap_start: datetime = gap_timestamps.min()  # type: ignore[assignment]
        gap_end: datetime = gap_timestamps.max()  # type: ignore[assignment]
        buffer_start = gap_start - timedelta(milliseconds=tf_ms * buffer_candles)
        buffer_end = gap_end + timedelta(milliseconds=tf_ms)

        db_pdf = query_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start=buffer_start,
            end=buffer_end,
        )

        if db_pdf.empty:
            logger.debug(f"[Reconciler] No DB data for feature recomputation {symbol} {timeframe}")
            return

        db_pdf = db_pdf.copy()
        db_pdf["timestamp"] = pd.to_datetime(db_pdf["timestamp"])

        db_pl = pl.from_pandas(db_pdf).sort("timestamp")
        if "is_interpolated" not in db_pl.columns:
            db_pl = db_pl.with_columns(pl.lit(0.0).alias("is_interpolated"))

        gap_pl = pl.DataFrame({"timestamp": gap_timestamps})
        db_pl = db_pl.with_columns(
            pl.when(pl.col("timestamp").is_in(gap_pl["timestamp"].to_list()))
            .then(pl.lit(1.0))
            .otherwise(pl.col("is_interpolated"))
            .alias("is_interpolated")
        )

        window_pdf = db_pl.select(
            ["timestamp", "open", "high", "low", "close", "volume", "is_interpolated"]
        ).to_pandas()
        window_pdf["symbol"] = symbol
        window_pdf["timeframe"] = timeframe
        window_pdf["source"] = "indicators"
        window_pdf["ingested_at"] = pd.Timestamp.now("UTC")

        compute_and_store_features(window_pdf, symbol=symbol, timeframe=timeframe)

    async def reconcile_all(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                result = await self.reconcile(symbol, timeframe)
                results.append(result)
        return results


def load_config_from_yaml(path: str = "config.yaml") -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return validate_config(raw)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    config_path = "config.yaml"
    symbol = None
    timeframe = None
    args = argv[:]

    i = 0
    while i < len(args):
        if args[i] in ("--config", "-c") and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
            continue
        if args[i] in ("--symbol", "-s") and i + 1 < len(args):
            symbol = args[i + 1]
            i += 2
            continue
        if args[i] in ("--timeframe", "-t") and i + 1 < len(args):
            timeframe = args[i + 1]
            i += 2
            continue
        if args[i] in ("--help", "-h"):
            print("Usage: python -m crypto_trading_framework.reconciler [--config path] [--symbol SYMBOL] [--timeframe TF]")
            return 0
        i += 1

    try:
        config = load_config_from_yaml(config_path)
    except (OSError, yaml.YAMLError) as exc:
        logger.error("Gagal memuat konfigurasi: %s", exc)
        print(f"Gagal memuat konfigurasi: {exc}")
        return 1

    reconciler = DataReconciler(config)

    async def _run() -> int:
        if symbol and timeframe:
            await reconciler.reconcile(symbol, timeframe)
        else:
            await reconciler.reconcile_all()
        return 0

    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
