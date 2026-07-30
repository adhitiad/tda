"""
Tests for kill switch module.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crypto_trading_framework.core.kill_switch import clear, is_active, trigger


@pytest.fixture(autouse=True)
def _clean_kill_switch():
    kill_file = Path("KILL_SWITCH.txt")
    if kill_file.exists():
        kill_file.unlink()
    yield
    if kill_file.exists():
        kill_file.unlink()


def test_is_active_false_when_no_kill_switch():
    assert not is_active()


def test_is_active_true_with_file():
    Path("KILL_SWITCH.txt").write_text("1", encoding="utf-8")
    assert is_active()


def test_is_active_true_with_file_case_insensitive():
    Path("KILL_SWITCH.txt").write_text("True", encoding="utf-8")
    assert is_active()


def test_is_active_true_with_redis(monkeypatch, tmp_path):
    monkeypatch.delenv("TIMESCALE_SERVICE_URL", raising=False)

    mock_cache = MagicMock()
    mock_cache.get.return_value = "1"

    with (
        patch("crypto_trading_framework.core.kill_switch.REDIS_AVAILABLE", True),
        patch("crypto_trading_framework.core.kill_switch.get_redis_cache", return_value=mock_cache),
    ):
        assert is_active()


def test_trigger_creates_file():
    trigger(reason="test")
    assert Path("KILL_SWITCH.txt").exists()
    assert Path("KILL_SWITCH.txt").read_text(encoding="utf-8") == "test"


def test_trigger_sets_redis_key(monkeypatch):
    mock_cache = MagicMock()

    with (
        patch("crypto_trading_framework.core.kill_switch.REDIS_AVAILABLE", True),
        patch("crypto_trading_framework.core.kill_switch.get_redis_cache", return_value=mock_cache),
    ):
        trigger(reason="redis_test")
        mock_cache.set.assert_called_once_with("kill_switch", "redis_test", ttl=3600)


def test_clear_removes_file():
    Path("KILL_SWITCH.txt").write_text("1", encoding="utf-8")
    clear()
    assert not Path("KILL_SWITCH.txt").exists()


def test_clear_deletes_redis_key(monkeypatch):
    mock_cache = MagicMock()

    with (
        patch("crypto_trading_framework.core.kill_switch.REDIS_AVAILABLE", True),
        patch("crypto_trading_framework.core.kill_switch.get_redis_cache", return_value=mock_cache),
    ):
        clear()
        mock_cache.delete.assert_called_once_with("kill_switch")
