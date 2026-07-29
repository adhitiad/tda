"""
Tests for walk-forward optimization framework.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from crypto_trading_framework.ml.backtest import Backtester
from crypto_trading_framework.walk_forward import (
    FoldResult,
    WalkForwardConfig,
    WalkForwardOptimizer,
    WalkForwardResult,
)


def _make_df(rows=300):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(rows) * 0.5)
    high = close + np.random.rand(rows) * 2
    low = close - np.random.rand(rows) * 2
    prob = np.clip(np.random.beta(2, 5, rows), 0, 1)
    atr = np.random.rand(rows) * 2
    timestamps = [datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(rows)]
    return pl.DataFrame({
        "timestamp": timestamps,
        "close": close,
        "high": high,
        "low": low,
        "prob": prob,
        "atr": atr,
    })


def _dummy_train(df, feature_cols):
    pass


class TestWalkForwardConfig:
    def test_defaults(self):
        cfg = WalkForwardConfig()
        assert cfg.n_splits == 5
        assert cfg.test_size == 50
        assert cfg.embargo == 0
        assert cfg.expanding is True
        assert cfg.train_size is None

    def test_custom(self):
        cfg = WalkForwardConfig(n_splits=3, test_size=20, embargo=5, expanding=False, train_size=100)
        assert cfg.n_splits == 3
        assert cfg.test_size == 20
        assert cfg.embargo == 5
        assert cfg.expanding is False
        assert cfg.train_size == 100


class TestWalkForwardOptimizer:
    def test_run_returns_result(self):
        df = _make_df(300)
        bt = Backtester(tp_pct=0.03, sl_pct=0.015, initial_capital=10000.0)
        optimizer = WalkForwardOptimizer(bt, WalkForwardConfig(n_splits=3, test_size=30))
        result = optimizer.run(df, threshold=0.8)
        assert isinstance(result, WalkForwardResult)
        assert len(result.folds) == 3

    def test_run_empty_df_raises(self):
        bt = Backtester()
        optimizer = WalkForwardOptimizer(bt, WalkForwardConfig())
        with pytest.raises(ValueError):
            optimizer.run(pl.DataFrame())

    def test_run_too_few_rows_raises(self):
        df = _make_df(50)
        bt = Backtester()
        optimizer = WalkForwardOptimizer(bt, WalkForwardConfig(n_splits=5, test_size=50))
        with pytest.raises(ValueError):
            optimizer.run(df, threshold=0.8)

    def test_run_with_train_func(self):
        df = _make_df(300)
        bt = Backtester(tp_pct=0.03, sl_pct=0.015, initial_capital=10000.0)
        optimizer = WalkForwardOptimizer(
            bt,
            WalkForwardConfig(n_splits=3, test_size=30),
            train_func=_dummy_train,
        )
        result = optimizer.run(df, threshold=0.8)
        assert len(result.folds) == 3

    def test_fold_result_fields(self):
        df = _make_df(300)
        bt = Backtester(tp_pct=0.03, sl_pct=0.015, initial_capital=10000.0)
        optimizer = WalkForwardOptimizer(bt, WalkForwardConfig(n_splits=2, test_size=30))
        result = optimizer.run(df, threshold=0.8)
        fold = result.folds[0]
        assert isinstance(fold, FoldResult)
        assert fold.fold_index == 0
        assert fold.train_rows > 0
        assert fold.test_rows == 30
        assert "total_trades" in fold.backtest_result

    def test_aggregate_metrics(self):
        df = _make_df(300)
        bt = Backtester(tp_pct=0.03, sl_pct=0.015, initial_capital=10000.0)
        optimizer = WalkForwardOptimizer(bt, WalkForwardConfig(n_splits=3, test_size=30))
        result = optimizer.run(df, threshold=0.8)
        assert isinstance(result.avg_winrate, float)
        assert isinstance(result.avg_profit_factor, float)
        assert isinstance(result.avg_return_pct, float)
        assert isinstance(result.avg_max_drawdown_pct, float)
        assert result.total_trades >= 0
        assert len(result.fold_returns) == 3
        assert len(result.fold_winrates) == 3

    def test_rolling_window(self):
        df = _make_df(300)
        bt = Backtester(tp_pct=0.03, sl_pct=0.015, initial_capital=10000.0)
        cfg = WalkForwardConfig(n_splits=3, test_size=30, expanding=False, train_size=100)
        optimizer = WalkForwardOptimizer(bt, cfg)
        result = optimizer.run(df, threshold=0.8)
        assert len(result.folds) == 3
        for fold in result.folds:
            assert fold.train_rows <= 100 + 30

    def test_embargo(self):
        df = _make_df(300)
        bt = Backtester(tp_pct=0.03, sl_pct=0.015, initial_capital=10000.0)
        cfg = WalkForwardConfig(n_splits=2, test_size=30, embargo=10)
        optimizer = WalkForwardOptimizer(bt, cfg)
        result = optimizer.run(df, threshold=0.8)
        assert len(result.folds) == 2
        train_end = datetime.fromisoformat(result.folds[0].train_end)
        test_start = datetime.fromisoformat(result.folds[0].test_start)
        assert (test_start - train_end).total_seconds() >= 10 * 3600

    def test_print_report(self):
        df = _make_df(300)
        bt = Backtester(tp_pct=0.03, sl_pct=0.015, initial_capital=10000.0)
        optimizer = WalkForwardOptimizer(bt, WalkForwardConfig(n_splits=2, test_size=30))
        result = optimizer.run(df, threshold=0.8)
        optimizer.print_report(result)
        assert True

    def test_consistency_score(self):
        df = _make_df(300)
        bt = Backtester(tp_pct=0.03, sl_pct=0.015, initial_capital=10000.0)
        optimizer = WalkForwardOptimizer(bt, WalkForwardConfig(n_splits=3, test_size=30))
        result = optimizer.run(df, threshold=0.8)
        assert 0.0 <= result.consistency_score <= 100.0
