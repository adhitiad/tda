"""
Portfolio-level risk management.

Tracks positions, computes correlation matrix from OHLCV returns,
applies risk-based position sizing, and enforces portfolio-level
drawdown circuit breaker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("portfolio")


@dataclass
class PortfolioRiskConfig:
    """Configuration for portfolio risk management."""

    model_config: dict | None = None
    enabled: bool = True
    max_portfolio_exposure: float = 0.5
    max_correlation_exposure: float = 0.3
    max_portfolio_drawdown_pct: float = 0.25
    lookback_correlation: int = 100
    risk_per_trade: float = 0.02
    rebalance_threshold: float = 0.1


@dataclass
class Position:
    """Active portfolio position."""

    symbol: str
    direction: str
    entry_price: float
    size: float
    entry_time: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


class PortfolioRiskManager:
    """Portfolio-level risk manager."""

    def __init__(self, config: PortfolioRiskConfig):
        self.config = config
        self.positions: dict[str, Position] = {}
        self.capital: float = 0.0
        self.peak_capital: float = 0.0
        self.correlation_matrix: pl.DataFrame | None = None

    def update_capital(self, capital: float):
        """Update portfolio capital and peak for drawdown tracking."""
        self.capital = capital
        self.peak_capital = max(self.peak_capital, capital)

    def add_position(self, position: Position):
        """Add or update a position."""
        self.positions[position.symbol] = position
        logger.info(f"[Portfolio] Added position {position.symbol}: {position.direction} size={position.size}")

    def remove_position(self, symbol: str):
        """Remove a closed position."""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"[Portfolio] Removed position {symbol}")

    def update_position(self, symbol: str, current_price: float):
        """Update unrealized PnL for an open position."""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        pos.current_price = current_price
        if pos.direction == "LONG":
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
        else:
            pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        atr: float,
        feature_history: pl.DataFrame | None = None,
    ) -> float:
        """
        Calculate position size based on portfolio risk budgeting.

        Uses inverse-volatility weighting when correlation data is available,
        otherwise falls back to equal risk per trade.

        :param symbol: Target symbol
        :param price: Current price
        :param atr: Average True Range for stop-loss calculation
        :param feature_history: Optional OHLCV history for correlation
        :return: Position size in base units
        """
        if not self.config.enabled or self.capital <= 0 or atr <= 0:
            return 0.0

        risk_amount = self.capital * self.config.risk_per_trade
        stop_distance = atr * 2
        size = risk_amount / stop_distance

        exposure = self.get_total_exposure()
        if exposure + (size * price) / self.capital > self.config.max_portfolio_exposure:
            available = self.config.max_portfolio_exposure - exposure
            size = (available * self.capital) / stop_distance

        if feature_history is not None and "close" in feature_history.columns:
            corr = self._check_symbol_correlation(symbol, feature_history)
            if corr > 0.7:
                size *= 0.5
                logger.info(f"[Portfolio] Reduced size for {symbol}: correlation={corr:.2f}")

        return max(size, 0.0)

    def get_total_exposure(self) -> float:
        """Get total portfolio exposure as fraction of capital."""
        if self.capital <= 0:
            return 0.0
        return sum(abs(p.size * p.current_price) for p in self.positions.values()) / self.capital

    def check_portfolio_drawdown(self) -> tuple[bool, float]:
        """
        Check if portfolio drawdown exceeds circuit breaker threshold.

        :return: (breached, drawdown_pct)
        """
        if self.peak_capital <= 0:
            return False, 0.0
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        breached = drawdown >= self.config.max_portfolio_drawdown_pct
        if breached:
            logger.warning(f"[Portfolio] Drawdown breach: {drawdown:.2%} >= {self.config.max_portfolio_drawdown_pct:.2%}")
        return breached, drawdown

    def update_correlation_matrix(self, ohlcv_data: dict[str, pl.DataFrame]):
        """
        Update correlation matrix from OHLCV data.

        :param ohlcv_data: Dict of symbol -> OHLCV DataFrame
        """
        if not self.config.enabled or not ohlcv_data:
            return
        returns = {}
        lookback = self.config.lookback_correlation
        for symbol, df in ohlcv_data.items():
            if "close" in df.columns and df.height > 1:
                close = df.select("close").tail(lookback)
                ret = close.with_columns(
                    pl.col("close").pct_change().alias(symbol)
                ).select(symbol)
                returns[symbol] = ret[symbol]
        if len(returns) >= 2:
            min_len = min(len(v) for v in returns.values())
            aligned = {k: v.head(min_len).drop_nulls() for k, v in returns.items()}
            if all(len(v) > 1 for v in aligned.values()):
                self.correlation_matrix = pl.DataFrame(aligned).corr()
                logger.info(f"[Portfolio] Correlation matrix updated for {len(aligned)} symbols")

    def _check_symbol_correlation(self, symbol: str, feature_history: pl.DataFrame) -> float:
        """
        Check correlation between symbol and existing positions.

        :param symbol: Target symbol
        :param feature_history: OHLCV history for target symbol
        :return: Max correlation with existing positions, or 0.0
        """
        if self.correlation_matrix is None or symbol not in self.correlation_matrix.columns:
            return 0.0
        existing_symbols = [s for s in self.positions if s in self.correlation_matrix.columns]
        if not existing_symbols:
            return 0.0
        col_series = self.correlation_matrix[symbol]
        correlations = []
        for existing in existing_symbols:
            if existing in self.correlation_matrix.columns:
                row_idx = self.correlation_matrix.columns.index(existing)
                corr_val = col_series[row_idx]
                if corr_val is not None and not np.isnan(corr_val):
                    correlations.append(abs(float(corr_val)))
        return max(correlations) if correlations else 0.0

    def should_rebalance(self) -> bool:
        """
        Check if portfolio should be rebalanced based on drift from target weights.
        """
        if not self.config.enabled or not self.positions:
            return False
        total = sum(abs(p.size * p.current_price) for p in self.positions.values())
        if total <= 0:
            return False
        target = 1.0 / len(self.positions)
        for pos in self.positions.values():
            weight = abs(pos.size * pos.current_price) / total
            if abs(weight - target) > self.config.rebalance_threshold:
                return True
        return False

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary for logging/alerting."""
        total_exposure = self.get_total_exposure()
        breached, drawdown = self.check_portfolio_drawdown()
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return {
            "capital": self.capital,
            "peak_capital": self.peak_capital,
            "total_exposure": total_exposure,
            "drawdown_pct": drawdown,
            "drawdown_breach": breached,
            "unrealized_pnl": unrealized,
            "open_positions": len(self.positions),
            "symbols": list(self.positions.keys()),
        }
