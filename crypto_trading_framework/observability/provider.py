"""
Observability data provider for the Local Observability Cockpit.

Reads from TimescaleDB, Redis, model registry JSON, and backtest artifacts
to feed the Streamlit dashboard.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from crypto_trading_framework.core.logging import get_logger
from crypto_trading_framework.db.database import query_ohlcv
from crypto_trading_framework.db.ledger import Ledger

try:
    from redis.exceptions import RedisError

    REDIS_ERROR_AVAILABLE = True
except ImportError:
    REDIS_ERROR_AVAILABLE = False

logger = get_logger("observability")

_DATA_ERRORS: tuple[type[BaseException], ...] = (OSError, ValueError, AttributeError, KeyError, TypeError, IndexError)
if REDIS_ERROR_AVAILABLE:
    _REDIS_ERRORS: tuple[type[BaseException], ...] = (RedisError,) + _DATA_ERRORS
else:
    _REDIS_ERRORS = _DATA_ERRORS

try:
    from crypto_trading_framework.db.redis_cache import get_redis_cache

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from crypto_trading_framework.ml.feature_store import get_features

    FEATURE_STORE_AVAILABLE = True
except ImportError:
    FEATURE_STORE_AVAILABLE = False

try:
    from crypto_trading_framework.ml.model_registry import (
        ModelRegistry,
        ModelRegistryConfig,
    )

    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False


class ObservabilityDataProvider:
    """Aggregates data from multiple sources for the cockpit dashboard."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._ledger = Ledger()
        self._registry = None
        if REGISTRY_AVAILABLE:
            try:
                reg_cfg = ModelRegistryConfig(
                    registry_dir=str(Path(self.config.get("paths", {}).get("models_dir", "models")) / "registry")
                )
                self._registry = ModelRegistry(reg_cfg)
            except _DATA_ERRORS as exc:
                logger.debug("[Observability] Gagal inisialisasi ModelRegistry: %s", exc)

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "-").replace(":", "")

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pl.DataFrame:
        normalized = self._normalize_symbol(symbol)
        try:
            pdf = query_ohlcv(symbol=normalized, timeframe=timeframe, source="ccxt", limit=limit)
            if pdf.empty:
                return pl.DataFrame()
            return pl.from_pandas(pdf).sort("timestamp")
        except _DATA_ERRORS as exc:
            logger.debug("[Observability] Gagal load OHLCV %s %s: %s", symbol, timeframe, exc)
            return pl.DataFrame()

    def get_indicators(self, symbol: str, timeframe: str, limit: int = 500) -> pl.DataFrame:
        if not FEATURE_STORE_AVAILABLE:
            return pl.DataFrame()
        normalized = self._normalize_symbol(symbol)
        try:
            df = get_features(symbol=normalized, timeframe=timeframe)
            if df.empty:
                return pl.DataFrame()
            df = df.sort_values("timestamp").tail(limit)
            return pl.from_pandas(df)
        except _DATA_ERRORS as exc:
            logger.debug("[Observability] Gagal load indikator %s %s: %s", symbol, timeframe, exc)
            return pl.DataFrame()

    def get_signals(self) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        if not REDIS_AVAILABLE:
            return signals
        try:
            cache = get_redis_cache()
            symbols = self.config.get("data", {}).get("symbols", [])
            timeframes = self.config.get("data", {}).get("timeframes", [])
            for symbol in symbols:
                for timeframe in timeframes:
                    key = f"signal:{symbol}:{timeframe}"
                    data = cache.get(key)
                    if data is not None:
                        sig = data if isinstance(data, dict) else {}
                        sig.setdefault("symbol", symbol)
                        sig.setdefault("timeframe", timeframe)
                        sig.setdefault("fetched_at", datetime.now(UTC).isoformat())
                        signals.append(sig)
        except _REDIS_ERRORS as exc:
            logger.debug("[Observability] Gagal baca sinyal dari Redis: %s", exc)
        return signals

    def get_model_registry(self) -> list[dict[str, Any]]:
        if not self._registry:
            return []
        try:
            if not hasattr(self._registry, "_index") or not self._registry._index:
                return []
            return [v for v in self._registry._index.get("models", {}).values()]
        except _DATA_ERRORS as exc:
            logger.debug("[Observability] Gagal baca model registry: %s", exc)
            return []

    def get_shadow_trader_performance(self) -> dict[str, Any]:
        try:
            return self._ledger.get_performance()
        except _DATA_ERRORS as exc:
            logger.debug("[Observability] Gagal baca performa shadow trader: %s", exc)
            return {}

    def get_open_trades(self) -> list[dict[str, Any]]:
        try:
            return self._ledger.get_open_trades()
        except _DATA_ERRORS as exc:
            logger.debug("[Observability] Gagal baca open trades: %s", exc)
            return []

    def get_closed_trades(self, limit: int = 100) -> list[dict[str, Any]]:
        try:
            trades = self._ledger.get_closed_trades()
            return trades[-limit:] if trades else []
        except _DATA_ERRORS as exc:
            logger.debug("[Observability] Gagal baca closed trades: %s", exc)

    def get_backtest_results(self, symbol: str | None = None, timeframe: str | None = None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        backtest_dir = Path(self.config.get("paths", {}).get("backtest_dir", "backtests"))
        if not backtest_dir.exists():
            return results
        try:
            for path in sorted(backtest_dir.glob("*.json"), reverse=True):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    data["_filename"] = path.name
                    if symbol and symbol not in data.get("symbol", ""):
                        continue
                    if timeframe and timeframe not in path.name:
                        continue
                    results.append(data)
                except _DATA_ERRORS:
                    continue
        except _DATA_ERRORS as exc:
            logger.debug("[Observability] Gagal baca backtest results: %s", exc)
        return results[:20]

    def get_latest_backtest(self, symbol: str, timeframe: str) -> dict[str, Any] | None:
        results = self.get_backtest_results(symbol, timeframe)
        return results[0] if results else None

    def get_smart_money_snapshot(self, symbol: str) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "composite_signal": "NEUTRAL",
            "veto_status": "NONE",
            "cex_signal": "NEUTRAL",
            "oi_signal": "NEUTRAL",
            "liq_signal": "NEUTRAL",
            "total_stablecoin_balance": 0.0,
            "oi_change_24h_pct": 0.0,
            "long_liquidation_usd": 0.0,
            "short_liquidation_usd": 0.0,
        }
        if not REDIS_AVAILABLE:
            return snapshot
        try:
            cache = get_redis_cache()
            tf = self.config.get("data", {}).get("timeframes", ["h1"])[0]
            key = f"signal:{symbol}:{tf}"
            data = cache.get(key)
            if isinstance(data, dict):
                sm = data.get("smart_money_analysis", {})
                if sm:
                    snapshot["composite_signal"] = sm.get("composite_signal", "NEUTRAL")
                    snapshot["veto_status"] = sm.get("veto_status", "NONE")
                    snapshot["cex_signal"] = sm.get("cex_signal", "NEUTRAL")
                    snapshot["oi_signal"] = sm.get("oi_signal", "NEUTRAL")
                    snapshot["liq_signal"] = sm.get("liq_signal", "NEUTRAL")
        except _REDIS_ERRORS as exc:
            logger.debug("[Observability] Gagal baca smart money snapshot: %s", exc)
        return snapshot

    def get_db_health(self) -> dict[str, Any]:
        try:
            from sqlalchemy import text

            from crypto_trading_framework.db.database import get_engine

            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "healthy", "dialect": engine.dialect.name}
        except _DATA_ERRORS as exc:
            logger.debug("[Observability] DB health check failed: %s", exc)
            return {"status": "unhealthy", "error": str(exc)}

    def get_redis_health(self) -> dict[str, Any]:
        if not REDIS_AVAILABLE:
            return {"status": "unavailable"}
        try:
            cache = get_redis_cache()
            result = cache.health_check()
            return result if isinstance(result, dict) else {"status": "unknown"}
        except _REDIS_ERRORS as exc:
            logger.debug("[Observability] Redis health check failed: %s", exc)
            return {"status": "unhealthy", "error": str(exc)}
