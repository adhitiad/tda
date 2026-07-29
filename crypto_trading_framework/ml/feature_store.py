"""
Feature store for offline/online feature consistency.

Stores computed indicators in TimescaleDB with point-in-time correctness,
and serves features from Redis cache during inference.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("feature_store")


# ---------------------------------------------------------------------------
# Base ORM model
# ---------------------------------------------------------------------------
class FeatureStoreBase(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Feature columns (mirrors add_all_indicators output)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "close", "volume",
    "ema_20", "ema_50", "ema_200",
    "bb_middle", "bb_upper", "bb_lower", "bb_width",
    "rsi", "stoch_k", "stoch_d",
    "atr",
    "macd", "macd_signal", "macd_hist",
    "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b", "ichimoku_chikou",
    "volume_profile_p20", "volume_profile_p50", "volume_profile_p80", "distance_from_poc",
    "volume_profile_skew", "volume_profile_kurtosis",
    "volatility", "autocorrelation", "kurtosis",
    "tick_imbalance", "trade_velocity", "spread_dynamics",
    "returns_1d", "returns_5d",
    "vwap", "vwap_deviation",
    "obv", "obv_change",
    "volume_spike", "volume_spike_flag",
    "regime_trending", "adx", "plus_di", "minus_di",
]


# ---------------------------------------------------------------------------
# ORM Model
# ---------------------------------------------------------------------------
class FeatureStore(FeatureStoreBase):
    __tablename__ = "feature_store"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    symbol: str = Column(String(64), nullable=False)
    timeframe: str = Column(String(16), nullable=False)
    timestamp: datetime = Column(DateTime, nullable=False)
    source: str = Column(String(32), nullable=False, default="indicators")
    ingested_at: datetime = Column(DateTime, nullable=False)

    close: float = Column(Float, nullable=True)
    volume: float = Column(Float, nullable=True)
    ema_20: float = Column(Float, nullable=True)
    ema_50: float = Column(Float, nullable=True)
    ema_200: float = Column(Float, nullable=True)
    bb_middle: float = Column(Float, nullable=True)
    bb_upper: float = Column(Float, nullable=True)
    bb_lower: float = Column(Float, nullable=True)
    bb_width: float = Column(Float, nullable=True)
    rsi: float = Column(Float, nullable=True)
    stoch_k: float = Column(Float, nullable=True)
    stoch_d: float = Column(Float, nullable=True)
    atr: float = Column(Float, nullable=True)
    macd: float = Column(Float, nullable=True)
    macd_signal: float = Column(Float, nullable=True)
    macd_hist: float = Column(Float, nullable=True)
    ichimoku_tenkan: float = Column(Float, nullable=True)
    ichimoku_kijun: float = Column(Float, nullable=True)
    ichimoku_senkou_a: float = Column(Float, nullable=True)
    ichimoku_senkou_b: float = Column(Float, nullable=True)
    ichimoku_chikou: float = Column(Float, nullable=True)
    volume_profile_p20: float = Column(Float, nullable=True)
    volume_profile_p50: float = Column(Float, nullable=True)
    volume_profile_p80: float = Column(Float, nullable=True)
    distance_from_poc: float = Column(Float, nullable=True)
    volume_profile_skew: float = Column(Float, nullable=True)
    volume_profile_kurtosis: float = Column(Float, nullable=True)
    volatility: float = Column(Float, nullable=True)
    autocorrelation: float = Column(Float, nullable=True)
    kurtosis: float = Column(Float, nullable=True)
    tick_imbalance: float = Column(Float, nullable=True)
    trade_velocity: float = Column(Float, nullable=True)
    spread_dynamics: float = Column(Float, nullable=True)
    returns_1d: float = Column(Float, nullable=True)
    returns_5d: float = Column(Float, nullable=True)
    vwap: float = Column(Float, nullable=True)
    vwap_deviation: float = Column(Float, nullable=True)
    obv: float = Column(Float, nullable=True)
    obv_change: float = Column(Float, nullable=True)
    volume_spike: float = Column(Float, nullable=True)
    volume_spike_flag: float = Column(Float, nullable=True)
    regime_trending: float = Column(Float, nullable=True)
    adx: float = Column(Float, nullable=True)
    plus_di: float = Column(Float, nullable=True)
    minus_di: float = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_feature_symbol_tf_ts"),
        Index("ix_feature_symbol_timeframe", "symbol", "timeframe"),
        Index("ix_feature_timestamp", "timestamp"),
    )


# ---------------------------------------------------------------------------
# Engine / Session factory
# ---------------------------------------------------------------------------
_engine = None
_SessionFactory = None


def _get_engine():
    global _engine
    if _engine is None:
        url = os.getenv("TIMESCALE_SERVICE_URL", "")
        if not url:
            raise ValueError("TIMESCALE_SERVICE_URL environment variable is required.")
        kwargs: dict[str, Any] = {"future": True}
        if not url.startswith("sqlite"):
            kwargs.update({
                "pool_size": 5,
                "max_overflow": 10,
                "pool_recycle": 1800,
            })
        _engine = create_engine(url, **kwargs)
    return _engine


def _get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=_get_engine(), autoflush=False, autocommit=False, future=True)
    return _SessionFactory


def _get_session() -> Session:
    return _get_session_factory()()


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------
def create_feature_store_schema():
    engine = _get_engine()
    FeatureStoreBase.metadata.create_all(engine)
    dialect = engine.dialect.name
    if dialect == "timescaledb":
        with _get_session() as session:
            session.execute(text("""
                SELECT create_hypertable('feature_store', 'timestamp', if_not_exists => TRUE);
            """))


def drop_feature_store_schema():
    engine = _get_engine()
    FeatureStoreBase.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# Feature store operations
# ---------------------------------------------------------------------------
def compute_and_store_features(df: pd.DataFrame, symbol: str, timeframe: str, source: str = "indicators") -> int:
    """
    Compute features from OHLCV DataFrame and store in feature store.
    This ensures offline features are computed once and reused.
    """
    if df.empty:
        return 0

    import polars as pl

    from crypto_trading_framework.core.indicators import add_all_indicators

    df_pl = pl.from_pandas(df)
    df_with_features = add_all_indicators(df_pl)

    available_feature_cols = [c for c in FEATURE_COLUMNS if c in df_with_features.columns]
    pdf = df_with_features.select(["timestamp"] + available_feature_cols).to_pandas()
    pdf["symbol"] = symbol
    pdf["timeframe"] = timeframe
    pdf["source"] = source
    pdf["ingested_at"] = datetime.utcnow()

    rows = 0
    with _get_session() as session:
        for _, row in pdf.iterrows():
            ts = row["timestamp"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            ingested = row["ingested_at"]
            if hasattr(ingested, "to_pydatetime"):
                ingested = ingested.to_pydatetime()

            values = {
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "timestamp": ts,
                "source": row["source"],
                "ingested_at": ingested,
            }
            for col in FEATURE_COLUMNS:
                if col in row:
                    values[col] = float(row[col]) if pd.notna(row[col]) else None

            cols_sql = ", ".join(values.keys())
            placeholders = ", ".join([f":{k}" for k in values])
            update_clause = ", ".join([f"{k} = EXCLUDED.{k}" for k in values if k not in ("symbol", "timeframe", "timestamp")])

            stmt = text(f"""
                INSERT INTO feature_store ({cols_sql})
                VALUES ({placeholders})
                ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                    {update_clause}
            """)
            session.execute(stmt, values)
            rows += 1
        session.commit()

    return rows


def get_features(symbol: str, timeframe: str, start: datetime | None = None, end: datetime | None = None, source: str = "indicators") -> pd.DataFrame:
    """
    Get features from store with optional time range.
    Supports point-in-time correctness via start/end bounds.
    """
    params: dict[str, Any] = {"symbol": symbol, "timeframe": timeframe, "source": source}
    where = ["symbol = :symbol", "timeframe = :timeframe", "source = :source"]

    if start:
        where.append("timestamp >= :start")
        params["start"] = start
    if end:
        where.append("timestamp <= :end")
        params["end"] = end

    sql = f"""
        SELECT timestamp, {", ".join(FEATURE_COLUMNS)}
        FROM feature_store
        WHERE {' AND '.join(where)}
        ORDER BY timestamp ASC
    """

    with _get_session() as session:
        result = session.execute(text(sql), params)
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=["timestamp"] + FEATURE_COLUMNS)
        return pd.DataFrame(rows, columns=["timestamp"] + FEATURE_COLUMNS)


def get_latest_features(symbol: str, timeframe: str, n_rows: int = 200, source: str = "indicators") -> pd.DataFrame | None:
    """
    Get latest N rows of features. Used during online inference.
    Checks Redis cache first, falls back to DB.
    """
    cache_key = f"features:{symbol}:{timeframe}:{source}:latest:{n_rows}"

    try:
        from crypto_trading_framework.db.redis_cache import get_redis_cache
        cache = get_redis_cache()
        cached = cache.get(cache_key)
        if cached is not None and hasattr(cached, "empty") and not cached.empty:
            return cached
    except Exception as e:
        logger.warning(f"[FeatureStore] Cache miss: {e}")

    df = get_features(symbol, timeframe, source=source)
    if df.empty:
        return None

    df = df.tail(n_rows)

    try:
        cache.set(cache_key, df, ttl=60)
    except Exception as e:
        logger.warning(f"[FeatureStore] Gagal cache features: {e}")

    return df


def invalidate_feature_cache(symbol: str, timeframe: str, source: str = "indicators"):
    """Invalidate Redis cache for features when new data arrives."""
    try:
        from crypto_trading_framework.db.redis_cache import get_redis_cache
        cache = get_redis_cache()
        pattern = f"features:{symbol}:{timeframe}:{source}:*"
        keys = cache._client.keys(pattern)
        if keys:
            cache._client.delete(*keys)
    except Exception as e:
        logger.warning(f"[FeatureStore] Gagal invalidate cache: {e}")
