"""
Tests for portfolio risk management.
"""

import numpy as np
import polars as pl
import pytest

from crypto_trading_framework.portfolio import (
    PortfolioRiskConfig,
    PortfolioRiskManager,
    Position,
)


@pytest.fixture
def portfolio():
    return PortfolioRiskManager(PortfolioRiskConfig())


class TestPortfolioRiskConfig:
    def test_defaults(self):
        cfg = PortfolioRiskConfig()
        assert cfg.enabled is True
        assert cfg.max_portfolio_exposure == 0.5
        assert cfg.max_portfolio_drawdown_pct == 0.25
        assert cfg.risk_per_trade == 0.02

    def test_custom(self):
        cfg = PortfolioRiskConfig(enabled=False, max_portfolio_exposure=0.3)
        assert cfg.enabled is False
        assert cfg.max_portfolio_exposure == 0.3


class TestPortfolioRiskManager:
    def test_add_and_remove_position(self, portfolio):
        pos = Position(symbol="BTC/USDT", direction="LONG", entry_price=100.0, size=1.0, entry_time="2023-01-01")
        portfolio.add_position(pos)
        assert "BTC/USDT" in portfolio.positions
        portfolio.remove_position("BTC/USDT")
        assert "BTC/USDT" not in portfolio.positions

    def test_update_position_pnl_long(self, portfolio):
        pos = Position(symbol="BTC/USDT", direction="LONG", entry_price=100.0, size=1.0, entry_time="2023-01-01")
        portfolio.add_position(pos)
        portfolio.update_position("BTC/USDT", 110.0)
        assert portfolio.positions["BTC/USDT"].unrealized_pnl == 10.0

    def test_update_position_pnl_short(self, portfolio):
        pos = Position(symbol="BTC/USDT", direction="SHORT", entry_price=100.0, size=1.0, entry_time="2023-01-01")
        portfolio.add_position(pos)
        portfolio.update_position("BTC/USDT", 90.0)
        assert portfolio.positions["BTC/USDT"].unrealized_pnl == 10.0

    def test_get_total_exposure(self, portfolio):
        portfolio.update_capital(10000.0)
        pos = Position(symbol="BTC/USDT", direction="LONG", entry_price=100.0, size=1.0, entry_time="2023-01-01", current_price=100.0)
        portfolio.add_position(pos)
        assert portfolio.get_total_exposure() == 0.01

    def test_check_portfolio_drawdown_no_breach(self, portfolio):
        portfolio.update_capital(10000.0)
        portfolio.update_capital(9000.0)
        breached, drawdown = portfolio.check_portfolio_drawdown()
        assert breached is False
        assert drawdown == 0.1

    def test_check_portfolio_drawdown_breach(self, portfolio):
        portfolio.update_capital(10000.0)
        portfolio.update_capital(7000.0)
        breached, drawdown = portfolio.check_portfolio_drawdown()
        assert breached is True
        assert drawdown == 0.3

    def test_calculate_position_size(self, portfolio):
        portfolio.update_capital(10000.0)
        size = portfolio.calculate_position_size("BTC/USDT", price=100.0, atr=2.0)
        assert size > 0.0
        expected = 10000.0 * 0.02 / (2.0 * 2)
        assert abs(size - expected) < 1e-9

    def test_calculate_position_size_disabled(self):
        portfolio = PortfolioRiskManager(PortfolioRiskConfig(enabled=False))
        portfolio.update_capital(10000.0)
        size = portfolio.calculate_position_size("BTC/USDT", price=100.0, atr=2.0)
        assert size == 0.0

    def test_calculate_position_size_exposure_limit(self, portfolio):
        portfolio.update_capital(10000.0)
        pos = Position(symbol="ETH/USDT", direction="LONG", entry_price=100.0, size=40.0, entry_time="2023-01-01", current_price=100.0)
        portfolio.add_position(pos)
        size = portfolio.calculate_position_size("BTC/USDT", price=100.0, atr=2.0)
        assert size < 500.0

    def test_should_rebalance(self, portfolio):
        portfolio.update_capital(10000.0)
        pos = Position(symbol="BTC/USDT", direction="LONG", entry_price=100.0, size=1.0, entry_time="2023-01-01", current_price=100.0)
        portfolio.add_position(pos)
        assert portfolio.should_rebalance() is False

    def test_get_portfolio_summary(self, portfolio):
        portfolio.update_capital(10000.0)
        summary = portfolio.get_portfolio_summary()
        assert "capital" in summary
        assert "drawdown_pct" in summary
        assert "open_positions" in summary


class TestCorrelationMatrix:
    def test_update_correlation_matrix(self, portfolio):
        np.random.seed(42)
        n = 100
        btc = 100 + np.cumsum(np.random.randn(n) * 0.5)
        eth = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df_btc = pl.DataFrame({"close": btc})
        df_eth = pl.DataFrame({"close": eth})
        portfolio.update_correlation_matrix({"BTC/USDT": df_btc, "ETH/USDT": df_eth})
        assert portfolio.correlation_matrix is not None

    def test_update_correlation_matrix_empty(self, portfolio):
        portfolio.update_correlation_matrix({})
        assert portfolio.correlation_matrix is None

    def test_update_correlation_matrix_single(self, portfolio):
        df = pl.DataFrame({"close": [100, 101, 102]})
        portfolio.update_correlation_matrix({"BTC/USDT": df})
        assert portfolio.correlation_matrix is None

    def test_check_symbol_correlation_high(self, portfolio):
        np.random.seed(42)
        n = 100
        btc = 100 + np.cumsum(np.random.randn(n) * 0.5)
        eth = btc + np.random.randn(n) * 0.01
        df_btc = pl.DataFrame({"close": btc})
        df_eth = pl.DataFrame({"close": eth})
        portfolio.update_correlation_matrix({"BTC/USDT": df_btc, "ETH/USDT": df_eth})
        pos = Position(symbol="BTC/USDT", direction="LONG", entry_price=100.0, size=1.0, entry_time="2023-01-01")
        portfolio.add_position(pos)
        corr = portfolio._check_symbol_correlation("ETH/USDT", df_eth)
        assert corr > 0.7

    def test_check_symbol_correlation_no_matrix(self, portfolio):
        corr = portfolio._check_symbol_correlation("ETH/USDT", pl.DataFrame({"close": [1, 2, 3]}))
        assert corr == 0.0
