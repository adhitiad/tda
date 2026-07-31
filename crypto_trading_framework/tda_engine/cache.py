from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import polars as pl

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("tda_cache")


class TDAFeatureCache:
    """Cache for Topological Data Analysis persistence diagrams.

    Avoids recomputing expensive persistence diagrams on every candle by
    using a sliding window with incremental update and file-based persistence.

    Parameters
    ----------
    cache_dir : str
        Directory for on-disk cache files.
    window_size : int
        Number of candles in the sliding window for diagram computation.
    update_interval_minutes : int
        Minimum minutes between recomputations for the same symbol/timeframe.
    max_cache_entries : int
        Maximum number of cached diagrams in memory.
    """

    def __init__(
        self,
        cache_dir: str = "cache/tda",
        window_size: int = 200,
        update_interval_minutes: int = 60,
        max_cache_entries: int = 100,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.update_interval_minutes = update_interval_minutes
        self.max_cache_entries = max_cache_entries
        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._access_times: dict[str, float] = {}

    def _cache_key(self, symbol: str, timeframe: str, window_hash: str) -> str:
        raw = f"{symbol}:{timeframe}:{window_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _window_hash(self, df: pl.DataFrame) -> str:
        """Hash the last N rows to detect if the window changed."""
        tail = df.tail(min(self.window_size, len(df)))
        data_bytes = tail.write_buffer().to_pybytes()
        return hashlib.md5(data_bytes).hexdigest()

    def _diagram_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, symbol: str, timeframe: str, df: pl.DataFrame) -> dict[str, Any] | None:
        """Retrieve a cached persistence diagram if the window hasn't changed."""
        if len(df) < self.window_size:
            return None

        window_hash = self._window_hash(df)
        key = self._cache_key(symbol, timeframe, window_hash)

        if key in self._memory_cache:
            self._access_times[key] = time.time()
            return self._memory_cache[key]

        path = self._diagram_path(key)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._memory_cache[key] = data
                self._access_times[key] = time.time()
                logger.debug(f"[TDA Cache] HIT for {symbol}/{timeframe}")
                return data
            except (OSError, json.JSONDecodeError):
                logger.debug(f"[TDA Cache] Corrupt cache file for {key}")

        return None

    def put(self, symbol: str, timeframe: str, df: pl.DataFrame, diagram: dict[str, Any]) -> None:
        """Store a persistence diagram in the cache."""
        if len(df) < self.window_size:
            return

        window_hash = self._window_hash(df)
        key = self._cache_key(symbol, timeframe, window_hash)

        self._memory_cache[key] = diagram
        self._access_times[key] = time.time()

        path = self._diagram_path(key)
        try:
            path.write_text(json.dumps(diagram, default=str), encoding="utf-8")
        except OSError as exc:
            logger.warning(f"[TDA Cache] Failed to write cache file: {exc}")

        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Evict least-recently-used entries when cache exceeds max size."""
        while len(self._memory_cache) > self.max_cache_entries:
            oldest_key = min(self._access_times, key=self._access_times.get)
            self._memory_cache.pop(oldest_key, None)
            self._access_times.pop(oldest_key, None)
            path = self._diagram_path(oldest_key)
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

    def invalidate(self, symbol: str | None = None, timeframe: str | None = None) -> None:
        """Invalidate cache entries, optionally filtered by symbol/timeframe."""
        keys_to_remove = []
        for key in self._memory_cache:
            if symbol and symbol not in key:
                continue
            if timeframe and timeframe not in key:
                continue
            keys_to_remove.append(key)
        for key in keys_to_remove:
            self._memory_cache.pop(key, None)
            self._access_times.pop(key, None)
            path = self._diagram_path(key)
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

    def cleanup_stale(self, max_age_minutes: int = 1440) -> int:
        """Remove cache entries older than max_age_minutes. Returns count removed."""
        now = time.time()
        removed = 0
        stale_keys = [k for k, t in self._access_times.items() if (now - t) > max_age_minutes * 60]
        for key in stale_keys:
            self._memory_cache.pop(key, None)
            self._access_times.pop(key, None)
            path = self._diagram_path(key)
            try:
                if path.exists():
                    path.unlink()
                    removed += 1
            except OSError:
                pass
        return removed

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "memory_entries": len(self._memory_cache),
            "max_entries": self.max_cache_entries,
            "cache_dir": str(self.cache_dir),
            "window_size": self.window_size,
            "update_interval_minutes": self.update_interval_minutes,
        }
