import asyncio
from datetime import datetime

import ccxt.async_support as ccxt_async
import pandas as pd
import polars as pl
import yfinance as yf

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("data_ingestion")

try:
    from crypto_trading_framework.db.database import (
        get_last_ingested_at,
        update_ingestion_state,
        upsert_ohlcv_dataframe,
    )
    from crypto_trading_framework.db.redis_cache import get_redis_cache
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    from crypto_trading_framework.core.validation import create_validator_from_config
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False


class DataIngestion:
    """Mengambil data OHLCV dari berbagai sumber (ccxt + yfinance) dan
    mengonversi langsung ke Polars DataFrame untuk pemrosesan cepat."""

    TIMEFRAME_MAP = {
        "m5": "5m",
        "m15": "15m",
        "m30": "30m",
        "h1": "1h",
        "h4": "4h",
        "d1": "1d",
    }

    def __init__(self, exchange_id: str = "tokocrypto", symbol: str = "BTC/USDT"):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.exchange = getattr(ccxt_async, exchange_id)()

    async def fetch_ohlcv_async(
        self, timeframe: str, limit: int = 2000
    ) -> pl.DataFrame | None:
        try:
            all_ohlcv = []
            tf_ms = self.exchange.parse_timeframe(timeframe) * 1000 if hasattr(self.exchange, "parse_timeframe") else 60000
            now = self.exchange.milliseconds()
            since = max(0, now - (limit * tf_ms))
            fetch_limit = 300

            while len(all_ohlcv) < limit:
                batch = await self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, since=since, limit=fetch_limit)
                if not batch:
                    break
                since = batch[-1][0] + 1
                all_ohlcv.extend(batch)
                if len(batch) < fetch_limit:
                    break

            if not all_ohlcv:
                all_ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)

            if not all_ohlcv:
                return None

            df = pl.DataFrame(
                all_ohlcv,
                schema=["timestamp", "open", "high", "low", "close", "volume"],
                orient="row",
            )
            df = df.unique(subset=["timestamp"]).sort("timestamp")
            df = df.with_columns(
                pl.col("timestamp").cast(pl.Datetime("ms"))
            )
            return df
        except Exception as e:
            logger.error(f"Gagal fetch {timeframe}: {e}")
            return None

    async def fetch_orderbook_async(self, limit: int = 100) -> pl.DataFrame | None:
        try:
            ob = await self.exchange.fetch_order_book(self.symbol, limit=limit)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            if not bids or not asks:
                return None

            if len(bids) > 0 and len(bids[0]) >= 2:
                bid_prices = [float(b[0]) for b in bids]
                bid_sizes = [float(b[1]) for b in bids]
            else:
                return None

            if len(asks) > 0 and len(asks[0]) >= 2:
                ask_prices = [float(a[0]) for a in asks]
                ask_sizes = [float(a[1]) for a in asks]
            else:
                return None

            bid_df = pl.DataFrame({"bid_price": bid_prices, "bid_size": bid_sizes})
            ask_df = pl.DataFrame({"ask_price": ask_prices, "ask_size": ask_sizes})

            bid_total = float(bid_df["bid_size"].sum())
            ask_total = float(ask_df["ask_size"].sum())
            total = bid_total + ask_total
            imbalance = (bid_total - ask_total) / total if total > 0.0 else 0.0

            best_bid = max(bid_prices)
            best_ask = min(ask_prices)
            spread = best_ask - best_bid
            spread_pct = spread / best_bid if best_bid > 0.0 else 0.0

            return pl.DataFrame({
                "timestamp": [pl.lit(datetime.now())],
                "bid_ask_imbalance": [imbalance],
                "spread": [spread],
                "spread_pct": [spread_pct],
                "bid_total": [bid_total],
                "ask_total": [ask_total],
                "best_bid": [best_bid],
                "best_ask": [best_ask],
            })
        except Exception as e:
            logger.error(f"Gagal fetch orderbook: {e}")
            return None

    async def fetch_funding_rate_async(self) -> pl.DataFrame | None:
        try:
            if hasattr(self.exchange, "fetch_funding_rate"):
                funding = await self.exchange.fetch_funding_rate(self.symbol)
                return pl.DataFrame({
                    "timestamp": [pl.lit(datetime.now())],
                    "funding_rate": [funding.get("fundingRate", 0.0)],
                })
        except Exception as e:
            logger.error(f"Gagal fetch funding rate: {e}")
        return None

    async def fetch_open_interest_async(self) -> pl.DataFrame | None:
        try:
            if hasattr(self.exchange, "fetch_open_interest"):
                oi = await self.exchange.fetch_open_interest(self.symbol)
                return pl.DataFrame({
                    "timestamp": [pl.lit(datetime.now())],
                    "open_interest": [oi.get("openInterest", 0.0)],
                })
        except Exception as e:
            logger.error(f"Gagal fetch open interest: {e}")
        return None

    async def fetch_multi_timeframe(
        self, timeframes: list[str] | None = None, limit: int = 500
    ) -> dict[str, pl.DataFrame]:
        if timeframes is None:
            timeframes = list(self.TIMEFRAME_MAP.keys())

        tasks = [self.fetch_ohlcv_async(self.TIMEFRAME_MAP[tf], limit) for tf in timeframes]
        results = await asyncio.gather(*tasks)

        data = {}
        for tf, df in zip(timeframes, results):
            if df is not None and not df.is_empty():
                data[tf] = df
                logger.info(f"{tf}: {df.height} candles")

        return data

    def fetch_yfinance(
        self, ticker: str = "BTC-USD", period: str = "1y", interval: str = "1h"
    ) -> pl.DataFrame | None:
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period, interval=interval)

            if hist.empty:
                return None

            df = pl.from_pandas(hist.reset_index())

            rename_map = {}
            if "Datetime" in df.columns:
                rename_map["Datetime"] = "timestamp"
            elif "Date" in df.columns:
                rename_map["Date"] = "timestamp"

            for old, new in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")]:
                if old in df.columns:
                    rename_map[old] = new

            df = df.rename(rename_map)

            if "timestamp" in df.columns and df.schema["timestamp"] != pl.Datetime:
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

            return df
        except Exception as e:
            logger.error(f"Gagal fetch yfinance {ticker}: {e}")
            return None

    async def close(self):
        await self.exchange.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.run(self.close())

    def _normalize_symbol(self) -> str:
        return self.symbol.replace("/", "-").replace(":", "")

    def fill_missing_data(self, df: pl.DataFrame, method: str = "linear") -> pl.DataFrame:
        if df.is_empty():
            return df

        df_sorted = df.sort("timestamp")
        ts_col = "timestamp"
        numeric_cols = [c for c in df_sorted.columns if c != ts_col]

        if not numeric_cols:
            return df_sorted

        full_range = pl.date_range(
            low=df_sorted[ts_col].min(),
            high=df_sorted[ts_col].max(),
            interval="1m",
            eager=True,
        )
        full_df = pl.DataFrame({ts_col: full_range})
        merged = full_df.join(df_sorted, on=ts_col, how="left")

        for col in numeric_cols:
            if col not in merged.columns:
                continue
            if method == "forward_fill":
                merged = merged.with_columns(pl.col(col).forward_fill().alias(col))
            elif method == "linear":
                merged = merged.with_columns(pl.col(col).interpolate().alias(col))

        return merged

    def _validate_ohlcv(self, df: pl.DataFrame, source: str) -> pl.DataFrame | None:
        if not VALIDATION_AVAILABLE:
            return df

        try:
            validator = create_validator_from_config()
            result = validator.validate(df)

            if result.errors:
                for err in result.errors:
                    logger.error(f"[Validation] {source}: {err}")
                return None

            for warn in result.warnings:
                logger.warning(f"[Validation] {source}: {warn}")

            if not validator.allow_gaps:
                df = self.fill_missing_data(df, method="linear")

            return df
        except Exception as e:
            logger.error(f"[Validation] Gagal validasi data: {e}")
            return df

    def persist_to_db(self, df: pl.DataFrame | None, source: str = "ccxt", timeframe: str = "1h") -> int:
        if not DB_AVAILABLE or df is None or df.is_empty():
            return 0

        try:
            df = self._validate_ohlcv(df, source)
            if df is None or df.is_empty():
                return 0

            pdf = df.to_pandas()
            pdf["symbol"] = self._normalize_symbol()
            pdf["timeframe"] = timeframe
            rows = upsert_ohlcv_dataframe(pdf, source=source)

            last_ts = pdf["timestamp"].max()
            if pd.notna(last_ts):
                update_ingestion_state(
                    symbol=self._normalize_symbol(),
                    timeframe=timeframe,
                    source=source,
                    last_ingested_at=pd.Timestamp(last_ts).to_pydatetime(),
                    row_count=len(pdf),
                )

            logger.info(f"[DB] Persisted {rows} rows for {self.symbol} {timeframe} from {source}")
            return rows
        except Exception as e:
            logger.error(f"[DB] Gagal persist data: {e}")
            return 0

    def load_from_db(self, timeframe: str = "1h", source: str = "ccxt") -> pl.DataFrame | None:
        if not DB_AVAILABLE:
            return None

        try:
            from crypto_trading_framework.db.database import query_ohlcv
            pdf = query_ohlcv(
                symbol=self._normalize_symbol(),
                timeframe=timeframe,
                source=source,
            )
            if pdf.empty:
                return None
            return pl.from_pandas(pdf)
        except Exception as e:
            logger.error(f"[DB] Gagal load data: {e}")
            return None

    def cache_data(self, df: pl.DataFrame | None, timeframe: str = "1h") -> bool:
        if not DB_AVAILABLE or df is None or df.is_empty():
            return False

        try:
            cache = get_redis_cache()
            cache.cache_ohlcv(self._normalize_symbol(), timeframe, df.to_pandas())
            logger.info(f"[Cache] Cached {df.height} rows for {self.symbol} {timeframe}")
            return True
        except Exception as e:
            logger.error(f"[Cache] Gagal cache data: {e}")
            return False

    def get_cached_data(self, timeframe: str = "1h") -> pl.DataFrame | None:
        if not DB_AVAILABLE:
            return None

        try:
            cache = get_redis_cache()
            pdf = cache.get_ohlcv(self._normalize_symbol(), timeframe)
            if pdf is None:
                return None
            if hasattr(pdf, "empty") and pdf.empty:
                return None
            if hasattr(pdf, "is_empty") and pdf.is_empty():
                return None
            if isinstance(pdf, pl.DataFrame):
                return pdf
            return pl.from_pandas(pdf)
        except Exception as e:
            logger.error(f"[Cache] Gagal get cached data: {e}")
            return None

    async def fetch_ohlcv_with_persistence(
        self, timeframe: str, limit: int = 2000, source: str = "ccxt"
    ) -> pl.DataFrame | None:
        tf = self.TIMEFRAME_MAP.get(timeframe, timeframe)

        if DB_AVAILABLE:
            cached = self.get_cached_data(timeframe=tf)
            if cached is not None:
                logger.info(f"[Ingest] Using cached data for {self.symbol} {tf}")
                return cached

            last_ts = get_last_ingested_at(self._normalize_symbol(), tf, source)
            df = await self.fetch_ohlcv_async(tf, limit=limit)
            if df is not None and not df.is_empty():
                if last_ts is not None:
                    latest = pd.Timestamp(last_ts)
                    df = df.filter(pl.col("timestamp") > latest)
                if not df.is_empty():
                    self.persist_to_db(df, source=source, timeframe=tf)
                    self.cache_data(df, timeframe=tf)
                else:
                    logger.info(f"[Ingest] No new data for {self.symbol} {tf}")
            return df
        else:
            return await self.fetch_ohlcv_async(tf, limit=limit)

    def fetch_yfinance_with_persistence(
        self, ticker: str = "BTC-USD", period: str = "1y", interval: str = "1h", source: str = "yfinance"
    ) -> pl.DataFrame | None:
        symbol = self._normalize_symbol()
        if DB_AVAILABLE:
            cached = self.get_cached_data(timeframe=interval)
            if cached is not None:
                logger.info(f"[Ingest] Using cached data for {ticker} {interval}")
                return cached

        df = self.fetch_yfinance(ticker=ticker, period=period, interval=interval)
        if df is not None and not df.is_empty():
            df = self._validate_ohlcv(df, source)
            if df is not None and not df.is_empty():
                df["symbol"] = symbol
                df["timeframe"] = interval
                rows = upsert_ohlcv_dataframe(df.to_pandas(), source=source)
                cache = get_redis_cache()
                cache.cache_ohlcv(symbol, interval, df.to_pandas())
                logger.info(f"[Ingest] Persisted {rows} rows for {ticker} {interval}")
        return df
