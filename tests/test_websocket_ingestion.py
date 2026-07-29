"""
Tests for WebSocket streaming ingestion module.
"""

import json
from unittest.mock import AsyncMock, patch

import polars as pl
import pytest

from crypto_trading_framework.websocket_ingestion import (
    OKXWebSocketClient,
    WebSocketBuffer,
    start_websocket_ingestion,
)


class TestWebSocketBuffer:
    def test_empty_buffer_no_flush(self):
        buffer = WebSocketBuffer(flush_size=10, flush_interval=1.0)
        assert buffer.should_flush() is False

    def test_flush_by_size(self):
        buffer = WebSocketBuffer(flush_size=3, flush_interval=1.0)
        buffer.add({"timestamp": 1, "close": 100.0})
        buffer.add({"timestamp": 2, "close": 101.0})
        buffer.add({"timestamp": 3, "close": 102.0})
        assert buffer.should_flush() is True
        df = buffer.flush()
        assert df is not None
        assert len(df) == 3

    def test_flush_returns_dataframe(self):
        buffer = WebSocketBuffer(flush_size=10, flush_interval=1.0)
        buffer.add({"timestamp": 1700000000000, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0})
        df = buffer.flush()
        assert df is not None
        assert "timestamp" in df.columns
        assert df["timestamp"].dtype == pl.Datetime("ms")

    def test_flush_clears_buffer(self):
        buffer = WebSocketBuffer(flush_size=10, flush_interval=1.0)
        buffer.add({"timestamp": 1, "close": 100.0})
        buffer.flush()
        assert buffer.should_flush() is False


class TestOKXWebSocketClient:
    def test_ccxt_to_okx_inst_id(self):
        assert OKXWebSocketClient._ccxt_to_okx_inst_id("BTC/USDT:USDT") == "BTC-USDT"
        assert OKXWebSocketClient._ccxt_to_okx_inst_id("ETH/USDT:USDT") == "ETH-USDT"

    def test_okx_inst_id_to_normalized(self):
        assert OKXWebSocketClient._okx_inst_id_to_normalized("BTC-USDT") == "BTC/USDT"
        assert OKXWebSocketClient._okx_inst_id_to_normalized("ETH-USDT") == "ETH/USDT"

    def test_build_subscription_message(self):
        client = OKXWebSocketClient(
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            timeframes=["m15", "h1"],
            buffer=WebSocketBuffer(),
        )
        msg = json.loads(client._build_subscription_message())
        assert msg["op"] == "subscribe"
        assert len(msg["args"]) == 4
        args = { (a["instId"], a["channel"]) for a in msg["args"] }
        assert ("BTC-USDT", "candle15m") in args
        assert ("BTC-USDT", "candle1H") in args
        assert ("ETH-USDT", "candle15m") in args
        assert ("ETH-USDT", "candle1H") in args

    def test_parse_candle_message(self):
        data = ["1700000000000", "100.0", "101.0", "99.0", "100.5", "1000.0", "100000.0", "100500.0", "1"]
        candle = OKXWebSocketClient._parse_candle_message(data)
        assert candle == {
            "timestamp": 1700000000000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }

    def test_parse_candle_message_invalid(self):
        assert OKXWebSocketClient._parse_candle_message([]) is None
        assert OKXWebSocketClient._parse_candle_message([1, 2]) is None

    @pytest.mark.asyncio
    async def test_handle_message_adds_to_buffer(self):
        buffer = WebSocketBuffer(flush_size=100, flush_interval=1.0)
        client = OKXWebSocketClient(
            symbols=["BTC/USDT:USDT"],
            timeframes=["m15"],
            buffer=buffer,
        )
        msg = json.dumps({
            "arg": {"channel": "candle15m", "instId": "BTC-USDT"},
            "data": [["1700000000000", "100.0", "101.0", "99.0", "100.5", "1000.0", "100000.0", "100500.0", "1"]],
        })
        await client._handle_message(msg)
        assert len(buffer.buffer) == 1
        assert buffer.buffer[0]["symbol"] == "BTC/USDT"
        assert buffer.buffer[0]["timeframe"] == "m15"

    @pytest.mark.asyncio
    async def test_handle_message_flush_on_size(self):
        buffer = WebSocketBuffer(flush_size=2, flush_interval=1.0)
        on_flush = AsyncMock()
        client = OKXWebSocketClient(
            symbols=["BTC/USDT:USDT"],
            timeframes=["m15"],
            buffer=buffer,
            on_flush=on_flush,
        )
        msg1 = json.dumps({
            "arg": {"channel": "candle15m", "instId": "BTC-USDT"},
            "data": [["1700000000000", "100.0", "101.0", "99.0", "100.5", "1000.0", "100000.0", "100500.0", "1"]],
        })
        msg2 = json.dumps({
            "arg": {"channel": "candle15m", "instId": "BTC-USDT"},
            "data": [["1700000060000", "100.5", "101.5", "100.0", "101.0", "2000.0", "101000.0", "101500.0", "1"]],
        })
        await client._handle_message(msg1)
        assert not on_flush.called
        await client._handle_message(msg2)
        assert on_flush.called

    @pytest.mark.asyncio
    async def test_handle_message_ignores_subscribe_event(self):
        buffer = WebSocketBuffer()
        client = OKXWebSocketClient(
            symbols=["BTC/USDT:USDT"],
            timeframes=["m15"],
            buffer=buffer,
        )
        msg = json.dumps({"event": "subscribe"})
        await client._handle_message(msg)
        assert len(buffer.buffer) == 0


class TestStartWebSocketIngestion:
    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        with patch("crypto_trading_framework.websocket_ingestion.logger") as mock_logger:
            await start_websocket_ingestion({})
            mock_logger.info.assert_called_with("[WS] WebSocket ingestion dinonaktifkan")

    @pytest.mark.asyncio
    async def test_enabled_starts_client(self):
        config = {
            "websocket": {"enabled": True, "buffer_size": 10, "buffer_interval": 1.0},
            "data": {"symbols": ["BTC/USDT:USDT"], "timeframes": ["m15"]},
            "trading": {"max_symbols": 10},
        }
        with patch("crypto_trading_framework.websocket_ingestion.OKXWebSocketClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.start = AsyncMock()
            await start_websocket_ingestion(config)
            mock_client.start.assert_called_once()
