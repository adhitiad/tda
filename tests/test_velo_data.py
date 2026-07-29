"""
Tests for VeloDataClient.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from crypto_trading_framework.velo_data import VeloDataClient


@pytest.fixture
def mock_velo_client():
    mock = MagicMock()
    mock.timestamp.return_value = 1700000000000
    mock.get_rows.return_value = pd.DataFrame({
        "time": [1700000000000, 1700000060000],
        "exchange": ["binance", "binance"],
        "product": ["BTCUSDT", "BTCUSDT"],
        "buy_dollar_volume": [100.0, 200.0],
        "sell_dollar_volume": [50.0, 150.0],
    })
    mock.get_futures_columns.return_value = ["open_price", "close_price"]
    mock.get_futures.return_value = [{"exchange": "binance", "product": "BTCUSDT"}]
    return mock


def test_velo_client_init_with_env(monkeypatch):
    monkeypatch.setenv("VELO_API_KEY", "test_key")
    with patch("crypto_trading_framework.velo_data.velo") as mock_velo:
        mock_velo.client.return_value = MagicMock()
        client = VeloDataClient()
        assert client.api_key == "test_key"


def test_velo_client_init_with_arg():
    with patch("crypto_trading_framework.velo_data.velo") as mock_velo:
        mock_velo.client.return_value = MagicMock()
        client = VeloDataClient(api_key="arg_key")
        assert client.api_key == "arg_key"


def test_velo_client_init_missing_key(monkeypatch):
    monkeypatch.delenv("VELO_API_KEY", raising=False)
    with patch("crypto_trading_framework.velo_data.velo") as mock_velo:
        mock_velo.client.return_value = MagicMock()
        with pytest.raises(ValueError):
            VeloDataClient()


def test_get_futures_columns(mock_velo_client):
    with patch("crypto_trading_framework.velo_data.velo") as mock_velo:
        mock_velo.client.return_value = mock_velo_client
        client = VeloDataClient(api_key="test_key")
        cols = client.get_futures_columns()
        assert cols == ["open_price", "close_price"]


def test_get_cvd_data(mock_velo_client):
    with patch("crypto_trading_framework.velo_data.velo") as mock_velo:
        mock_velo.client.return_value = mock_velo_client
        client = VeloDataClient(api_key="test_key")
        df = client.get_cvd_data(
            exchanges=["binance"],
            products=["BTCUSDT"],
            hours=1,
            resolution="1m",
        )
        assert df is not None
        assert "time" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["time"])


def test_get_funding_rate(mock_velo_client):
    with patch("crypto_trading_framework.velo_data.velo") as mock_velo:
        mock_velo.client.return_value = mock_velo_client
        client = VeloDataClient(api_key="test_key")
        df = client.get_funding_rate(
            exchanges=["binance-futures"],
            coins=["BTC"],
            hours=1,
            resolution="1m",
        )
        assert df is not None
        assert "time" in df.columns


def test_get_open_interest(mock_velo_client):
    with patch("crypto_trading_framework.velo_data.velo") as mock_velo:
        mock_velo.client.return_value = mock_velo_client
        client = VeloDataClient(api_key="test_key")
        df = client.get_open_interest(
            exchanges=["binance-futures"],
            coins=["BTC"],
            hours=1,
            resolution="1m",
        )
        assert df is not None
        assert "time" in df.columns


def test_get_futures_basis(mock_velo_client):
    with patch("crypto_trading_framework.velo_data.velo") as mock_velo:
        mock_velo.client.return_value = mock_velo_client
        client = VeloDataClient(api_key="test_key")
        df = client.get_futures_basis(
            coins=["BTC", "ETH"],
            hours=24,
            resolution="1h",
        )
        assert df is not None
        assert "time" in df.columns


def test_get_orderbook_depth(mock_velo_client):
    mock_velo_client.depth.return_value = iter([
        pd.DataFrame({"timestamp": [1700000000000], "midprice": [50000.0]}),
    ])
    with patch("crypto_trading_framework.velo_data.velo") as mock_velo:
        mock_velo.client.return_value = mock_velo_client
        client = VeloDataClient(api_key="test_key")
        df = client.get_orderbook_depth(
            exchange="binance-futures",
            product="BTCUSDT",
            hours=1,
            resolution="5m",
        )
        assert df is not None
        assert "timestamp" in df.columns


def test_save_to_csv(tmp_path, mock_velo_client):
    with patch("crypto_trading_framework.velo_data.velo") as mock_velo:
        mock_velo.client.return_value = mock_velo_client
        client = VeloDataClient(api_key="test_key")
        df = client.get_cvd_data(
            exchanges=["binance"],
            products=["BTCUSDT"],
            hours=1,
            resolution="1m",
        )
        out_file = tmp_path / "test.csv"
        client.save_to_csv(df, str(out_file))
        assert out_file.exists()


def test_get_rows_returns_none(mock_velo_client):
    mock_velo_client.get_rows.return_value = None
    with patch("crypto_trading_framework.velo_data.velo") as mock_velo:
        mock_velo.client.return_value = mock_velo_client
        client = VeloDataClient(api_key="test_key")
        df = client.get_cvd_data(
            exchanges=["binance"],
            products=["BTCUSDT"],
            hours=1,
            resolution="1m",
        )
        assert df is None
