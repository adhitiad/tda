"""
Tests for REST API module.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from crypto_trading_framework.core.api import BotController, create_app


@pytest.fixture
def app():
    config = {
        "api": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8000,
            "api_key": "test-key",
            "allowed_origins": ["*"],
        },
        "data": {
            "exchange_id": "okx",
            "symbols": ["BTC/USDT:USDT"],
            "max_symbols": 10,
            "timeframes": ["m15", "h1"],
            "lookback": 2000,
        },
        "trading": {
            "dry_run": True,
            "max_symbols": 10,
        },
        "backtest": {
            "initial_capital": 10000.0,
            "max_drawdown_pct": 0.20,
        },
    }
    BotController._instance = None
    app = create_app(config)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_has_timestamp(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "timestamp" in data


class TestApiKeyAuth:
    def test_missing_api_key(self, app):
        client = TestClient(app)
        resp = client.post("/control/start")
        assert resp.status_code == 403

    def test_invalid_api_key(self, app):
        client = TestClient(app)
        resp = client.post("/control/start", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 403

    def test_valid_api_key(self, app):
        client = TestClient(app, headers={"X-API-Key": "test-key"})
        resp = client.get("/control/status")
        assert resp.status_code == 200


class TestControlEndpoints:
    def test_status_initial(self, client):
        resp = client.get("/control/status", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"

    def test_start_bot(self, client):
        with patch("crypto_trading_framework.api.AutomatedTradingBot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot_class.return_value = mock_bot
            resp = client.post("/control/start", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "started"
            assert "started_at" in data

    def test_stop_bot(self, app, client):
        with patch("crypto_trading_framework.api.AutomatedTradingBot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot.stop.return_value = None
            mock_bot_class.return_value = mock_bot
            client.post("/control/start", headers={"X-API-Key": "test-key"})
            resp = client.post("/control/stop", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "stopped"

    def test_double_start(self, client):
        with patch("crypto_trading_framework.api.AutomatedTradingBot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot_class.return_value = mock_bot
            client.post("/control/start", headers={"X-API-Key": "test-key"})
            resp = client.post("/control/start", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "already_running"


class TestSignalsEndpoint:
    def test_get_signals_empty(self, client):
        resp = client.get("/signals", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        assert resp.json()["signals"] == []

    def test_get_signals_with_models(self, client):
        controller = BotController.get_instance()
        controller.configure({
            "data": {"exchange_id": "okx", "symbols": ["BTC/USDT:USDT"], "max_symbols": 10, "timeframes": ["m15"], "lookback": 2000},
            "trading": {"dry_run": True, "max_symbols": 10},
            "backtest": {"initial_capital": 10000.0, "max_drawdown_pct": 0.20},
        })
        mock_model = MagicMock()
        controller.bot.trained_models = {
            "BTC/USDT": {
                "model": mock_model,
                "scaler": MagicMock(),
                "feature_cols": ["close", "volume"],
                "time_steps": 60,
            }
        }
        resp = client.get("/signals", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["signals"]) == 1
        assert data["signals"][0]["symbol"] == "BTC/USDT"


class TestModelsEndpoint:
    def test_get_models_empty(self, client):
        resp = client.get("/models", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        assert resp.json()["models"] == []

    def test_get_models_with_models(self, client):
        controller = BotController.get_instance()
        controller.configure({
            "data": {"exchange_id": "okx", "symbols": ["BTC/USDT:USDT"], "max_symbols": 10, "timeframes": ["m15"], "lookback": 2000},
            "trading": {"dry_run": True, "max_symbols": 10},
            "backtest": {"initial_capital": 10000.0, "max_drawdown_pct": 0.20},
        })
        mock_model = MagicMock()
        controller.bot.trained_models = {
            "BTC/USDT": {
                "model": mock_model,
                "scaler": MagicMock(),
                "feature_cols": ["close"],
                "time_steps": 60,
            }
        }
        resp = client.get("/models", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["symbol"] == "BTC/USDT"


class TestBacktestEndpoint:
    def test_backtest_missing_symbol(self, client):
        resp = client.post("/backtest", headers={"X-API-Key": "test-key"}, json={})
        assert resp.status_code == 422

    def test_backtest_queued(self, client):
        with patch("crypto_trading_framework.api.AutomatedTradingBot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot_class.return_value = mock_bot
            controller = BotController.get_instance()
            controller.configure({
                "data": {"exchange_id": "okx", "symbols": ["BTC/USDT:USDT"], "max_symbols": 10, "timeframes": ["m15"], "lookback": 2000},
                "trading": {"dry_run": True, "max_symbols": 10},
                "backtest": {"initial_capital": 10000.0, "max_drawdown_pct": 0.20},
            })
            resp = client.post("/backtest", headers={"X-API-Key": "test-key"}, json={"symbol": "BTC/USDT"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "queued"
