"""
Tests for alerting module.
"""

from unittest.mock import MagicMock, patch

import pytest

from crypto_trading_framework.core.alerting import (
    Alert,
    AlertManager,
    DiscordChannel,
    EmailChannel,
    TelegramChannel,
    alert_drawdown_breach,
    alert_execution_error,
    alert_model_drift,
    alert_training_failure,
)


class TestAlert:
    def test_str_with_context(self):
        a = Alert(level="ERROR", message="test", context={"k": "v"})
        assert str(a) == "[ERROR] test (k=v)"

    def test_str_without_context(self):
        a = Alert(level="INFO", message="hello")
        assert str(a) == "[INFO] hello"


class TestTelegramChannel:
    def test_send_success(self):
        channel = TelegramChannel(bot_token="TOKEN", chat_id="CHAT")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("crypto_trading_framework.alerting.urlopen", return_value=mock_resp):
            assert channel.send(Alert(level="INFO", message="hi")) is True

    def test_send_failure(self):
        channel = TelegramChannel(bot_token="TOKEN", chat_id="CHAT")
        with patch("crypto_trading_framework.alerting.urlopen", side_effect=OSError("fail")):
            assert channel.send(Alert(level="INFO", message="hi")) is False


class TestDiscordChannel:
    def test_send_success(self):
        channel = DiscordChannel(webhook_url="https://discord.com/api/webhooks/123")
        mock_resp = MagicMock()
        mock_resp.status = 204
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("crypto_trading_framework.alerting.urlopen", return_value=mock_resp):
            assert channel.send(Alert(level="INFO", message="hi")) is True

    def test_send_failure(self):
        channel = DiscordChannel(webhook_url="https://discord.com/api/webhooks/123")
        with patch("crypto_trading_framework.alerting.urlopen", side_effect=OSError("fail")):
            assert channel.send(Alert(level="INFO", message="hi")) is False


class TestEmailChannel:
    def test_send_success(self):
        channel = EmailChannel(
            smtp_host="smtp.example.com",
            smtp_port=465,
            username="user",
            password="pass",
            from_addr="from@example.com",
            to_addrs=["to@example.com"],
        )
        mock_server = MagicMock()
        with patch("crypto_trading_framework.alerting.smtplib.SMTP_SSL", return_value=mock_server):
            assert channel.send(Alert(level="INFO", message="hi")) is True

    def test_send_failure(self):
        channel = EmailChannel(
            smtp_host="smtp.example.com",
            smtp_port=465,
            username="user",
            password="pass",
            from_addr="from@example.com",
            to_addrs=["to@example.com"],
        )
        with patch("crypto_trading_framework.alerting.smtplib.SMTP_SSL", side_effect=OSError("fail")):
            assert channel.send(Alert(level="INFO", message="hi")) is False


class TestAlertManager:
    def setup_method(self):
        AlertManager._instance = None
        self.manager = AlertManager.get_instance()

    def test_configure_telegram(self):
        config = {
            "min_level": "INFO",
            "telegram": {"enabled": True, "bot_token": "T", "chat_id": "C"},
        }
        self.manager.configure(config)
        assert len(self.manager.channels) == 1
        assert isinstance(self.manager.channels[0], TelegramChannel)

    def test_configure_discord(self):
        config = {
            "min_level": "INFO",
            "discord": {"enabled": True, "webhook_url": "https://discord.com/api/webhooks/123"},
        }
        self.manager.configure(config)
        assert len(self.manager.channels) == 1
        assert isinstance(self.manager.channels[0], DiscordChannel)

    def test_configure_email(self):
        config = {
            "min_level": "INFO",
            "email": {
                "enabled": True,
                "smtp_host": "smtp.example.com",
                "smtp_port": 465,
                "from_addr": "from@example.com",
                "to_addrs": ["to@example.com"],
            },
        }
        self.manager.configure(config)
        assert len(self.manager.channels) == 1
        assert isinstance(self.manager.channels[0], EmailChannel)

    def test_send_no_channels(self):
        self.manager.configure({"min_level": "INFO"})
        assert self.manager.send(Alert(level="INFO", message="hi")) is False

    def test_send_min_level_filter(self):
        self.manager.configure({"min_level": "ERROR"})
        mock_channel = MagicMock()
        mock_channel.send.return_value = True
        self.manager.channels.append(mock_channel)
        assert self.manager.send(Alert(level="INFO", message="hi")) is False
        assert self.manager.send(Alert(level="ERROR", message="hi")) is True

    def test_send_success(self):
        config = {
            "min_level": "INFO",
            "telegram": {"enabled": True, "bot_token": "T", "chat_id": "C"},
        }
        self.manager.configure(config)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("crypto_trading_framework.alerting.urlopen", return_value=mock_resp):
            assert self.manager.send(Alert(level="INFO", message="hi")) is True


class TestHelperFunctions:
    def test_alert_training_failure(self):
        manager = AlertManager.get_instance()
        manager.channels = []
        manager.configure({"min_level": "INFO"})
        mock_channel = MagicMock()
        manager.channels.append(mock_channel)
        alert_training_failure("BTC/USDT", "OOM", {"epoch": 5})
        mock_channel.send.assert_called_once()
        a = mock_channel.send.call_args[0][0]
        assert a.level == "ERROR"
        assert "BTC/USDT" in a.message

    def test_alert_execution_error(self):
        manager = AlertManager.get_instance()
        manager.channels = []
        manager.configure({"min_level": "INFO"})
        mock_channel = MagicMock()
        manager.channels.append(mock_channel)
        alert_execution_error("ETH/USDT", "timeout", {"side": "buy"})
        a = mock_channel.send.call_args[0][0]
        assert a.level == "ERROR"
        assert "ETH/USDT" in a.message

    def test_alert_drawdown_breach(self):
        manager = AlertManager.get_instance()
        manager.channels = []
        manager.configure({"min_level": "INFO"})
        mock_channel = MagicMock()
        manager.channels.append(mock_channel)
        alert_drawdown_breach("BTC/USDT", 0.25, 0.20)
        a = mock_channel.send.call_args[0][0]
        assert a.level == "CRITICAL"
        assert "drawdown" in a.message.lower()

    def test_alert_model_drift(self):
        manager = AlertManager.get_instance()
        manager.channels = []
        manager.configure({"min_level": "INFO"})
        mock_channel = MagicMock()
        manager.channels.append(mock_channel)
        alert_model_drift("BTC/USDT", 0.15, 0.10)
        a = mock_channel.send.call_args[0][0]
        assert a.level == "WARNING"
        assert "drift" in a.message.lower()
