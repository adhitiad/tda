"""
Walk-forward optimization framework for time series backtesting.

Provides expanding/rolling window validation with embargoed test periods
and performance attribution per fold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from crypto_trading_framework.ml.backtest import Backtester, TradeResult
from crypto_trading_framework.core.logging import get_logger

logger = get_logger("walk_forward")


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""

    model_config: dict | None = None
    n_splits: int = 5
    train_size: int | None = None
    test_size: int = 50
    embargo: int = 0
    min_train_size: int = 100
    expanding: bool = True


@dataclass
class FoldResult:
    """Hasil backtest untuk satu fold."""

    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_rows: int
    test_rows: int
    backtest_result: dict
    trades: list[TradeResult]


@dataclass
class WalkForwardResult:
    """Agregasi hasil walk-forward optimization."""

    folds: list[FoldResult]
    config: WalkForwardConfig
    avg_winrate: float
    avg_profit_factor: float
    avg_return_pct: float
    avg_max_drawdown_pct: float
    total_trades: int
    consistency_score: float
    fold_returns: list[float]
    fold_winrates: list[float]


TrainFunc = Callable[[pl.DataFrame, list[str]], tuple[Any, Any]]


class WalkForwardOptimizer:
    """Orkestrator walk-forward optimization."""

    def __init__(
        self,
        backtester: Backtester,
        config: WalkForwardConfig,
        train_func: TrainFunc | None = None,
    ):
        self.backtester = backtester
        self.config = config
        self.train_func = train_func

    def run(self, df: pl.DataFrame, prob_col: str = "prob", threshold: float = 0.75) -> WalkForwardResult:
        """
        Menjalankan walk-forward optimization.

        :param df: DataFrame lengkap dengan fitur dan probabilitas
        :param prob_col: Kolom probabilitas prediksi
        :param threshold: Ambang batas untuk entry sinyal
        :return: WalkForwardResult dengan agregasi dan per-fold metrics
        """
        if df.is_empty():
            raise ValueError("DataFrame kosong tidak bisa dijalankan walk-forward")

        df_sorted = df.sort("timestamp")
        n_rows = len(df_sorted)
        config = self.config

        if config.n_splits <= 0:
            raise ValueError("n_splits harus lebih besar dari 0")

        total_test = config.n_splits * config.test_size
        if total_test > n_rows - config.min_train_size:
            raise ValueError(
                f"Data terlalu sedikit: butuh minimal {config.min_train_size + total_test} rows, "
                f"punya {n_rows}"
            )

        folds: list[FoldResult] = []
        fold_returns: list[float] = []
        fold_winrates: list[float] = []

        for i in range(config.n_splits):
            test_end = n_rows - i * config.test_size
            test_start = test_end - config.test_size
            embargo_end = test_start - config.embargo

            if config.expanding or config.train_size is None:
                train_end = max(config.min_train_size, embargo_end)
                train_start = 0
            else:
                window = config.train_size
                train_end = max(config.min_train_size, embargo_end)
                train_start = max(0, train_end - window)

            if train_end <= train_start:
                logger.warning(f"Fold {i} dilewati: data tidak cukup")
                continue

            train_df = df_sorted.slice(train_start, train_end - train_start)
            test_df = df_sorted.slice(test_start, test_end - test_start)

            train_timestamps = train_df["timestamp"].to_list()
            test_timestamps = test_df["timestamp"].to_list()
            train_start_ts = str(train_timestamps[0]) if train_timestamps else ""
            train_end_ts = str(train_timestamps[-1]) if train_timestamps else ""
            test_start_ts = str(test_timestamps[0]) if test_timestamps else ""
            test_end_ts = str(test_timestamps[-1]) if test_timestamps else ""

            if self.train_func is not None:
                feature_cols = [c for c in train_df.columns if c not in ("timestamp", "prob", "target")]
                self.train_func(train_df, feature_cols)

            bt_result = self.backtester.run_backtest(test_df, prob_col=prob_col, threshold=threshold)
            trades = bt_result.get("trades", [])

            fold = FoldResult(
                fold_index=i,
                train_start=train_start_ts,
                train_end=train_end_ts,
                test_start=test_start_ts,
                test_end=test_end_ts,
                train_rows=len(train_df),
                test_rows=len(test_df),
                backtest_result=bt_result,
                trades=trades,
            )
            folds.append(fold)
            fold_returns.append(bt_result.get("return_pct", 0.0))
            fold_winrates.append(bt_result.get("winrate", 0.0))

        if not folds:
            raise RuntimeError("Tidak ada fold yang berhasil dijalankan")

        avg_winrate = float(np.mean(fold_winrates)) if fold_winrates else 0.0
        avg_return = float(np.mean(fold_returns)) if fold_returns else 0.0
        avg_profit_factor = float(np.mean([f.backtest_result.get("profit_factor", 0.0) for f in folds]))
        avg_max_dd = float(np.mean([f.backtest_result.get("max_drawdown_pct", 0.0) for f in folds]))
        total_trades = sum(f.backtest_result.get("total_trades", 0) for f in folds)

        return_std = float(np.std(fold_returns)) if len(fold_returns) > 1 else 0.0
        consistency_score = max(0.0, 100.0 - return_std) if return_std > 0 else 100.0

        return WalkForwardResult(
            folds=folds,
            config=config,
            avg_winrate=round(avg_winrate, 2),
            avg_profit_factor=round(avg_profit_factor, 2),
            avg_return_pct=round(avg_return, 2),
            avg_max_drawdown_pct=round(avg_max_dd, 2),
            total_trades=total_trades,
            consistency_score=round(consistency_score, 2),
            fold_returns=[round(r, 2) for r in fold_returns],
            fold_winrates=[round(w, 2) for w in fold_winrates],
        )

    def print_report(self, result: WalkForwardResult, title: str = "WALK-FORWARD REPORT"):
        """Mencetak laporan walk-forward optimization."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(title)
        logger.info("=" * 80)
        logger.info(f"Folds             : {len(result.folds)}")
        logger.info(f"Total Trades      : {result.total_trades}")
        logger.info(f"Avg Winrate       : {result.avg_winrate}%")
        logger.info(f"Avg Return        : {result.avg_return_pct}%")
        logger.info(f"Avg Profit Factor : {result.avg_profit_factor}")
        logger.info(f"Avg Max Drawdown  : {result.avg_max_drawdown_pct}%")
        logger.info(f"Consistency Score : {result.consistency_score}%")
        logger.info("")
        logger.info("Per-Fold Returns : " + str(result.fold_returns))
        logger.info("Per-Fold Winrates: " + str(result.fold_winrates))
        logger.info("=" * 80)
        for fold in result.folds:
            bt = fold.backtest_result
            logger.info(
                f"Fold {fold.fold_index}: train=[{fold.train_start} -> {fold.train_end}], "
                f"test=[{fold.test_start} -> {fold.test_end}], "
                f"trades={bt.get('total_trades', 0)}, "
                f"return={bt.get('return_pct', 0)}%, "
                f"winrate={bt.get('winrate', 0)}%"
            )
        logger.info("=" * 80)
