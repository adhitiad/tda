"""
Redis caching layer for fast lookups.

Supports three deployment options:
1. Redis Cloud (redis.io)  -> standard redis:// or rediss://
2. Upstash (serverless Redis) -> rediss:// with Upstash endpoint
3. Local Redis instance -> redis://localhost:6379

All three are accessed through the same redis-py client interface.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from redis import Redis
from redis.connection import ConnectionPool


from crypto_trading_framework.core.logging import get_logger

logger = get_logger("redis_cache")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_redis_url(
    host: str = "localhost",
    port: int = 6379,
    password: str | None = None,
    db: int = 0,
    ssl: bool = False,
    username: str | None = None,
) -> str:
    scheme = "rediss" if ssl else "redis"
    auth = ""
    if username and password:
        auth = f"{username}:{password}@"
    elif password:
        auth = f":{password}@"
    return f"{scheme}://{auth}{host}:{port}/{db}"


def _build_redis_url_from_env() -> str:
    env_url = os.getenv("REDIS_URL")
    if env_url:
        return env_url

    provider = os.getenv("REDIS_PROVIDER", "local").lower()

    if provider == "redis_cloud":
        host = os.getenv("REDIS_HOST", "redis-12345.c123.us-east-1-2.ec2.cloud.redislabs.com")
        port = int(os.getenv("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD")
        username = os.getenv("REDIS_USERNAME")
        ssl = os.getenv("REDIS_SSL", "false").lower() == "true"
        return _build_redis_url(host=host, port=port, password=password, db=0, ssl=ssl, username=username)

    if provider == "upstash":
        host = os.getenv("REDIS_HOST", "us1-abc123.upstash.io")
        port = int(os.getenv("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD")
        ssl = True
        return _build_redis_url(host=host, port=port, password=password, db=0, ssl=ssl)

    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD")
    return _build_redis_url(host=host, port=port, password=password, db=0, ssl=False)


# ---------------------------------------------------------------------------
# Redis Cache Client
# ---------------------------------------------------------------------------
class RedisCache:
    def __init__(self, url: str | None = None, decode_responses: bool = True):
        self.url = url or _build_redis_url_from_env()
        self._pool = ConnectionPool.from_url(self.url, decode_responses=decode_responses, health_check_interval=0)
        self._client = Redis(connection_pool=self._pool)
        self._default_ttl = int(os.getenv("REDIS_DEFAULT_TTL", "3600"))

        try:
            self._client.ping()
            logger.info(f"[RedisCache] Connected to Redis: {self._mask_url(self.url)}")
        except Exception as e:
            logger.warning(f"[RedisCache] Failed to connect to Redis: {e}")

    @staticmethod
    def _mask_url(url: str) -> str:
        if "@" in url:
            parts = url.split("@")
            return "***@" + parts[1]
        return url

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize(value: Any) -> str:
        if isinstance(value, pd.DataFrame):
            return json.dumps({"__type": "dataframe", "data": value.to_json(orient="split")})
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value)
        return str(value)

    @staticmethod
    def _deserialize(value: str) -> Any:
        try:
            obj = json.loads(value)
            if isinstance(obj, dict) and obj.get("__type") == "dataframe":
                return pd.read_json(obj["data"], orient="split")
            return obj
        except (json.JSONDecodeError, ValueError):
            return value

    # ------------------------------------------------------------------
    # Basic operations
    # ------------------------------------------------------------------
    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        try:
            serialized = self._serialize(value)
            return self._client.set(key, serialized, ex=ttl or self._default_ttl)
        except Exception as e:
            logger.error(f"[RedisCache] SET error for key={key}: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        try:
            raw = self._client.get(key)
            if raw is None:
                return default
            return self._deserialize(raw)
        except Exception as e:
            logger.error(f"[RedisCache] GET error for key={key}: {e}")
            return default

    def delete(self, key: str) -> bool:
        try:
            return bool(self._client.delete(key))
        except Exception as e:
            logger.error(f"[RedisCache] DELETE error for key={key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.error(f"[RedisCache] EXISTS error for key={key}: {e}")
            return False

    def set_ttl(self, key: str, ttl: int) -> bool:
        try:
            return bool(self._client.expire(key, ttl))
        except Exception as e:
            logger.error(f"[RedisCache] EXPIRE error for key={key}: {e}")
            return False

    def flush_db(self) -> bool:
        try:
            return self._client.flushdb()
        except Exception as e:
            logger.error(f"[RedisCache] FLUSHDB error: {e}")
            return False

    # ------------------------------------------------------------------
    # Higher-level helpers
    # ------------------------------------------------------------------
    def cache_dataframe(self, key: str, df: pd.DataFrame, ttl: int | None = None) -> bool:
        if df is None or df.empty:
            return False
        return self.set(key, df, ttl=ttl)

    def get_dataframe(self, key: str) -> pd.DataFrame | None:
        value = self.get(key)
        if isinstance(value, pd.DataFrame):
            return value
        return None

    def cache_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame, ttl: int | None = None) -> str:
        key = f"ohlcv:{symbol}:{timeframe}"
        self.cache_dataframe(key, df, ttl=ttl)
        return key

    def get_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        key = f"ohlcv:{symbol}:{timeframe}"
        return self.get_dataframe(key)

    def cache_metadata(self, symbol: str, metadata: dict, ttl: int | None = None) -> str:
        key = f"meta:{symbol}"
        self.set(key, metadata, ttl=ttl)
        return key

    def get_metadata(self, symbol: str, default: dict | None = None) -> dict:
        key = f"meta:{symbol}"
        value = self.get(key)
        return value if isinstance(value, dict) else (default or {})

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Connection factories for each deployment option
# ---------------------------------------------------------------------------
def create_redis_cache(provider: str | None = None, url: str | None = None) -> RedisCache:
    """
    Create a RedisCache instance for a specific deployment option.

    :param provider: One of 'local', 'redis_cloud', 'upstash'. Defaults to REDIS_PROVIDER env var or 'local'.
    :param url: Optional explicit Redis URL. If provided, provider is ignored.
    :return: RedisCache instance
    """
    if url:
        return RedisCache(url=url)

    provider = (provider or os.getenv("REDIS_PROVIDER", "local")).lower()

    if provider == "redis_cloud":
        logger.info("[RedisCache] Using Redis Cloud deployment")
        return RedisCache()
    elif provider == "upstash":
        logger.info("[RedisCache] Using Upstash serverless Redis")
        return RedisCache()
    else:
        logger.info("[RedisCache] Using local Redis instance")
        return RedisCache()


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_cache_instance: RedisCache | None = None


def get_redis_cache() -> RedisCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = create_redis_cache()
    return _cache_instance


def reset_redis_cache():
    global _cache_instance
    if _cache_instance is not None:
        try:
            _cache_instance.close()
        except Exception:
            pass
    _cache_instance = None
