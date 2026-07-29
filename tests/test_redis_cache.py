"""
Tests for the Redis cache layer.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from crypto_trading_framework.db.redis_cache import (
    RedisCache,
    _build_redis_url,
    create_redis_cache,
    get_redis_cache,
)


def test_build_redis_url_local():
    url = _build_redis_url(host="localhost", port=6379)
    assert url == "redis://localhost:6379/0"


def test_build_redis_url_with_password():
    url = _build_redis_url(host="localhost", port=6379, password="secret")
    assert url == "redis://:secret@localhost:6379/0"


def test_build_redis_url_with_username_and_password():
    url = _build_redis_url(host="localhost", port=6379, password="secret", username="user")
    assert url == "redis://user:secret@localhost:6379/0"


def test_build_redis_url_ssl():
    url = _build_redis_url(host="localhost", port=6379, ssl=True)
    assert url == "rediss://localhost:6379/0"


@pytest.fixture
def mock_cache():
    with patch("crypto_trading_framework.redis_cache.Redis") as mock_redis:
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        cache = RedisCache(url="redis://localhost:6379/0")
        cache._client = mock_client
        yield cache, mock_client


def test_set_and_get(mock_cache):
    cache, client = mock_cache
    client.get.return_value = "hello"
    assert cache.get("test_key") == "hello"


def test_set_calls_client(mock_cache):
    cache, client = mock_cache
    cache.set("key", "value", ttl=60)
    client.set.assert_called_once()
    call_kwargs = client.set.call_args[1]
    assert call_kwargs.get("ex") == 60


def test_delete(mock_cache):
    cache, client = mock_cache
    client.delete.return_value = 1
    assert cache.delete("key") is True


def test_exists(mock_cache):
    cache, client = mock_cache
    client.exists.return_value = 1
    assert cache.exists("key") is True


def test_get_default(mock_cache):
    cache, client = mock_cache
    client.get.return_value = None
    assert cache.get("missing") is None
    assert cache.get("missing", default="fallback") == "fallback"


def test_cache_ohlcv(mock_cache):
    cache, client = mock_cache
    df = pd.DataFrame({"close": [100.0, 101.0]})
    client.set.return_value = True
    key = cache.cache_ohlcv("BTCUSDT", "1h", df, ttl=300)
    assert key == "ohlcv:BTCUSDT:1h"
    client.set.assert_called_once()


def test_cache_metadata(mock_cache):
    cache, client = mock_cache
    meta = {"last_price": 50000.0}
    client.set.return_value = True
    key = cache.cache_metadata("ETHUSDT", meta, ttl=600)
    assert key == "meta:ETHUSDT"
    client.set.assert_called_once()


def test_create_redis_cache_local(monkeypatch):
    monkeypatch.setenv("REDIS_PROVIDER", "local")
    with patch("crypto_trading_framework.redis_cache.Redis") as mock_redis:
        mock_redis.return_value = MagicMock()
        cache = create_redis_cache()
        assert isinstance(cache, RedisCache)


def test_create_redis_cache_explicit_url():
    with patch("crypto_trading_framework.redis_cache.Redis") as mock_redis:
        mock_redis.return_value = MagicMock()
        cache = create_redis_cache(url="redis://testhost:6379")
        assert isinstance(cache, RedisCache)


def test_singleton_get_redis_cache(monkeypatch):
    monkeypatch.setenv("REDIS_PROVIDER", "local")
    with patch("crypto_trading_framework.redis_cache.Redis") as mock_redis:
        mock_redis.return_value = MagicMock()
        cache1 = get_redis_cache()
        cache2 = get_redis_cache()
        assert cache1 is cache2
