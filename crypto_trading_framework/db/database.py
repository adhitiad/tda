"""
Database layer for TimescaleDB integration.

Provides SQLAlchemy engine factory, session management, hypertable setup,
and idempotent ingestion helpers for OHLCV and derived market data.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("database")


# ---------------------------------------------------------------------------
# Base ORM model
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------
class MarketDataOHLCV(Base):
    __tablename__ = "market_data_ohlcv"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="ccxt")
    is_interpolated: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_ohlcv_symbol_tf_ts"),
        Index("ix_ohlcv_symbol_timeframe", "symbol", "timeframe"),
        Index("ix_ohlcv_timestamp", "timestamp"),
    )


class IngestionState(Base):
    __tablename__ = "ingestion_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    last_ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "source", name="uq_ingestion_state_symbol_tf_src"),
    )


class Wallet(Base):
    __tablename__ = "wallet"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    initial_balance: Mapped[float] = mapped_column(Float, nullable=False)
    current_balance: Mapped[float] = mapped_column(Float, nullable=False)
    available_margin: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class TradeHistory(Base):
    __tablename__ = "trade_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    close_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    leverage: Mapped[str] = mapped_column(String(16), nullable=False)
    pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="OPEN")
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    atr: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    take_profit: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index("ix_trade_history_symbol", "symbol"),
        Index("ix_trade_history_status", "status"),
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DatabaseConfig:
    url: str
    pool_size: int = 5
    max_overflow: int = 10
    pool_recycle: int = 1800
    echo: bool = False


def get_database_config(override_url: str = "") -> DatabaseConfig:
    url = override_url or os.getenv("TIMESCALE_SERVICE_URL", "")
    if not url:
        raise ValueError(
            "Database URL is required. Set TIMESCALE_SERVICE_URL environment variable "
            "or provide database.url in config.yaml. "
            "Example: timescaledb://tsdbadmin:PASSWORD@host:31621/tsdb?sslmode=require"
        )
    return DatabaseConfig(url=url, echo=os.getenv("SQL_ECHO", "false").lower() == "true")


# ---------------------------------------------------------------------------
# Engine / Session factory
# ---------------------------------------------------------------------------
_engine = None
_SessionFactory = None
_db_url_override = ""


def get_engine(override_url: str = ""):
    global _engine, _db_url_override
    _db_url_override = override_url or _db_url_override
    if _engine is None:
        config = get_database_config(override_url=_db_url_override)
        kwargs: dict[str, Any] = {
            "echo": config.echo,
            "future": True,
        }
        if not config.url.startswith("sqlite"):
            kwargs.update({
                "pool_size": config.pool_size,
                "max_overflow": config.max_overflow,
                "pool_recycle": config.pool_recycle,
            })
        _engine = create_engine(config.url, **kwargs)
    return _engine


def get_session_factory(override_url: str = ""):
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(override_url=override_url), autoflush=False, autocommit=False, future=True)
    return _SessionFactory


def reset_database():
    """Reset engine/session factory (useful for tests)."""
    global _engine, _SessionFactory, _db_url_override
    _engine = None
    _SessionFactory = None
    _db_url_override = ""


def get_session(override_url: str = "") -> Session:
    return get_session_factory(override_url=override_url)()


@contextmanager
def session_scope(override_url: str = "") -> Generator[Session, None, None]:
    session = get_session(override_url=override_url)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------
def create_schema(override_url: str = ""):
    engine = get_engine(override_url=override_url)
    Base.metadata.create_all(engine)
    _enable_hypertables(override_url=override_url)
    _ensure_columns_exist(override_url=override_url)


def _ensure_columns_exist(override_url: str = ""):
    engine = get_engine(override_url=override_url)
    dialect = engine.dialect.name

    def _add_column(session, table: str, column: str, col_def: str):
        try:
            if dialect == "sqlite":
                result = session.execute(text(f"PRAGMA table_info({table})"))
                cols = [row[1] for row in result.fetchall()]
                if column not in cols:
                    session.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"))
            else:
                result = session.execute(text("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = :table AND column_name = :col
                """), {"table": table, "col": column})
                if not result.fetchone():
                    session.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"))
        except Exception as exc:
            logger.debug("Gagal memastikan kolom %s pada tabel %s: %s", column, table, exc)

    with session_scope(override_url=override_url) as session:
        _add_column(session, "market_data_ohlcv", "is_interpolated", "FLOAT NOT NULL DEFAULT 0.0")
        _add_column(session, "feature_store", "is_interpolated", "FLOAT NOT NULL DEFAULT 0.0")


def _enable_hypertables(override_url: str = ""):
    engine = get_engine(override_url=override_url)
    dialect = engine.dialect.name
    if dialect != "timescaledb":
        return
    with session_scope(override_url=override_url) as session:
        session.execute(text("""
            SELECT create_hypertable('market_data_ohlcv', 'timestamp', if_not_exists => TRUE);
        """))
        session.execute(text("""
            SELECT create_hypertable('ingestion_state', 'updated_at', if_not_exists => TRUE);
        """))


def drop_schema():
    engine = get_engine()
    Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# Idempotent ingestion helpers
# ---------------------------------------------------------------------------
def upsert_ohlcv_dataframe(df: pd.DataFrame, source: str = "ccxt") -> int:
    if df.empty:
        return 0

    required_cols = {"timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("ms")
    df["source"] = source
    df["updated_at"] = pd.Timestamp.now("UTC")
    if "is_interpolated" not in df.columns:
        df["is_interpolated"] = 0.0

    insert_cols = ["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume", "source", "updated_at", "is_interpolated"]
    records = df[insert_cols].to_dict("records")
    for rec in records:
        rec["timestamp"] = rec["timestamp"].to_pydatetime() if hasattr(rec["timestamp"], "to_pydatetime") else rec["timestamp"]
        rec["updated_at"] = rec["updated_at"].to_pydatetime() if hasattr(rec["updated_at"], "to_pydatetime") else rec["updated_at"]

    rows_inserted = 0
    with session_scope() as session:
        stmt = text("""
            INSERT INTO market_data_ohlcv
                (symbol, timeframe, timestamp, open, high, low, close, volume, source, updated_at, is_interpolated)
            VALUES
                (:symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume, :source, :updated_at, :is_interpolated)
            ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source,
                updated_at = EXCLUDED.updated_at,
                is_interpolated = EXCLUDED.is_interpolated
        """)
        for rec in records:
            result = session.execute(stmt, rec)
            if result.rowcount > 0:
                rows_inserted += result.rowcount

    return rows_inserted


def get_last_ingested_at(symbol: str, timeframe: str, source: str) -> datetime | None:
    with session_scope() as session:
        result = session.execute(text("""
            SELECT last_ingested_at FROM ingestion_state
            WHERE symbol = :symbol AND timeframe = :timeframe AND source = :source
        """), {"symbol": symbol, "timeframe": timeframe, "source": source})
        row = result.fetchone()
        return row[0] if row else None


def update_ingestion_state(symbol: str, timeframe: str, source: str, last_ingested_at: datetime, row_count: int):
    now = datetime.utcnow()
    with session_scope() as session:
        session.execute(text("""
            INSERT INTO ingestion_state (symbol, timeframe, source, last_ingested_at, row_count, updated_at)
            VALUES (:symbol, :timeframe, :source, :last_ingested_at, :row_count, :updated_at)
            ON CONFLICT (symbol, timeframe, source) DO UPDATE SET
                last_ingested_at = EXCLUDED.last_ingested_at,
                row_count = EXCLUDED.row_count,
                updated_at = EXCLUDED.updated_at
        """), {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": source,
            "last_ingested_at": last_ingested_at,
            "row_count": row_count,
            "updated_at": now,
        })


def get_data_range(symbol: str, timeframe: str, source: str = "ccxt") -> tuple[datetime | None, datetime | None]:
    with session_scope() as session:
        result = session.execute(text("""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM market_data_ohlcv
            WHERE symbol = :symbol AND timeframe = :timeframe AND source = :source
        """), {"symbol": symbol, "timeframe": timeframe, "source": source})
        row = result.fetchone()
        return (row[0], row[1]) if row and row[0] else (None, None)


def query_ohlcv(
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
    source: str = "ccxt",
    limit: int | None = None,
) -> pd.DataFrame:
    params: dict[str, Any] = {"symbol": symbol, "timeframe": timeframe, "source": source}
    where = ["symbol = :symbol", "timeframe = :timeframe", "source = :source"]
    if start:
        where.append("timestamp >= :start")
        params["start"] = start
    if end:
        where.append("timestamp <= :end")
        params["end"] = end

    sql = f"""
        SELECT timestamp, open, high, low, close, volume, source, is_interpolated
        FROM market_data_ohlcv
        WHERE {' AND '.join(where)}
        ORDER BY timestamp ASC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"

    with session_scope() as session:
        result = session.execute(text(sql), params)
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "source"])
        return pd.DataFrame(rows, columns=result.keys())
