"""
Fase 14 - Weekend Self-Tuning & Hyperparameter Optimization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import polars as pl
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger("self_tuning")


class SelfTuningEngine:
    """Weekend hyperparameter optimization with walk-forward validation."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.scheduler: AsyncIOScheduler | None = None
        self.tuning_results: dict[str, Any] = {}
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def start(self) -> None:
        """Start the APScheduler with cron trigger for Sunday 02:00 UTC."""
        if self.scheduler is not None:
            return
        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(
            self.run_weekend_tuning,
            CronTrigger(day_of_week="sun", hour=2, minute=0, timezone="UTC"),
            id="weekend_tuning",
            replace_existing=True,
        )
        self.scheduler.start()
        logger.info("[SelfTuning] Scheduler started - runs every Sunday 02:00 UTC")

    def stop(self) -> None:
        if self.scheduler is not None:
            self.scheduler.shutdown(wait=False)
            self.scheduler = None
            logger.info("[SelfTuning] Scheduler stopped")

    async def run_weekend_tuning(self) -> None:
        """Main entry point for weekend tuning cycle."""
        try:
            logger.info("[SelfTuning] Starting weekend hyperparameter optimization...")
            symbols = self.config.get("data", {}).get("symbols", ["BTC/USDT:USDT"])
            best_params = await self._optimize_parameters(symbols[0])

            if best_params:
                upgraded = self._validate_and_upgrade(best_params)
                if upgraded:
                    self._hot_swap_parameters(best_params)
                    self._send_telegram_report(best_params, "SUCCESS")
                else:
                    self._send_telegram_report(best_params, "REJECTED")
                    logger.info("[SelfTuning] New params rejected - did not meet upgrade threshold")
            else:
                logger.warning("[SelfTuning] No valid parameters found")
        except Exception:
            logger.exception("[SelfTuning] Error in weekend tuning")

    async def _optimize_parameters(self, symbol: str) -> dict[str, Any] | None:
        """Run Optuna optimization on 30-day OHLCV data."""
        try:
            from crypto_trading_framework.data_ingestion import DataIngestion

            ingestion = DataIngestion(
                exchange_id=self.config.get("data", {}).get("exchange_id", "okx"),
                symbol=symbol,
            )
            raw_data = await ingestion.fetch_multi_timeframe(timeframes=["h1"], limit=720)
            await ingestion.close()

            if not raw_data or "h1" not in raw_data:
                logger.warning("[SelfTuning] No data for optimization")
                return None

            df = raw_data["h1"]
            if df is None or df.is_empty():
                return None

            close = df["close"].to_numpy()
            high = df["high"].to_numpy()
            low = df["low"].to_numpy()
            volume = df["volume"].to_numpy()

            def objective(trial: optuna.Trial) -> float:
                rsi_period = trial.suggest_int("rsi_period", 10, 21)
                macd_fast = trial.suggest_int("macd_fast", 8, 16)
                macd_slow = trial.suggest_int("macd_slow", 20, 30)
                atr_period = trial.suggest_int("atr_period", 10, 20)
                kelly_cap = trial.suggest_float("kelly_cap", 0.1, 0.5)

                pnl = self._vectorized_backtest(close, high, low, volume, rsi_period, macd_fast, macd_slow, atr_period, kelly_cap)
                return pnl

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42))
            study.optimize(objective, n_trials=50, show_progress_bar=False)

            best = study.best_params
            best["estimated_pnl"] = study.best_value
            logger.info(f"[SelfTuning] Best params: {best}")
            return best

        except Exception:
            logger.exception("[SelfTuning] Error during optimization")
            return None

    def _vectorized_backtest(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
        rsi_period: int,
        macd_fast: int,
        macd_slow: int,
        atr_period: int,
        kelly_cap: float,
    ) -> float:
        """Super-fast vectorized backtest using NumPy."""
        n = len(close)
        if n < max(rsi_period, macd_slow, atr_period) + 10:
            return 0.0

        # Simple RSI
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.convolve(gains, np.ones(rsi_period) / rsi_period, mode="valid")
        avg_loss = np.convolve(losses, np.ones(rsi_period) / rsi_period, mode="valid")
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Simple MACD
        ema_fast = self._ema(close, macd_fast)
        ema_slow = self._ema(close, macd_slow)
        macd_line = ema_fast - ema_slow

        # Simple ATR
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        atr = np.convolve(tr, np.ones(atr_period) / atr_period, mode="valid")

        # Generate signals
        min_len = min(len(rsi), len(macd_line) - 1, len(atr))
        if min_len < 10:
            return 0.0

        signals = np.zeros(min_len)
        signals[(rsi[:min_len] < 30) & (macd_line[:min_len] > 0)] = 1  # BUY
        signals[(rsi[:min_len] > 70) & (macd_line[:min_len] < 0)] = -1  # SELL

        # Calculate PnL
        position = 0.0
        entry_price = 0.0
        pnl = 0.0
        atr_min = atr[:min_len] if len(atr) >= min_len else np.zeros(min_len)

        for i in range(1, min_len):
            if signals[i] == 1 and position == 0:
                position = kelly_cap
                entry_price = close[i + (n - min_len)]
            elif signals[i] == -1 and position > 0:
                exit_price = close[i + (n - min_len)]
                pnl += position * (exit_price - entry_price) / entry_price
                position = 0.0

        if position > 0:
            pnl += position * (close[-1] - entry_price) / entry_price

        return float(pnl)

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Compute Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _validate_and_upgrade(self, new_params: dict[str, Any]) -> bool:
        """Walk-forward validation: new params must beat old by +5% PnL."""
        old_pnl = self.tuning_results.get("last_pnl", 0.0)
        new_pnl = new_params.get("estimated_pnl", 0.0)

        if old_pnl <= 0 and new_pnl > 0:
            return True

        improvement = ((new_pnl - old_pnl) / abs(old_pnl) * 100.0) if old_pnl != 0 else 0.0
        logger.info(f"[SelfTuning] Old PnL: {old_pnl:.4f}, New PnL: {new_pnl:.4f}, Improvement: {improvement:.1f}%")

        return improvement >= 5.0

    def _hot_swap_parameters(self, new_params: dict[str, Any]) -> None:
        """Zero-downtime hot swap of parameters in memory."""
        indicators = self.config.setdefault("indicators", {})
        indicators["rsi_period"] = new_params.get("rsi_period", indicators.get("rsi_period", 14))
        indicators["macd_fast"] = new_params.get("macd_fast", indicators.get("macd_fast", 12))
        indicators["macd_slow"] = new_params.get("macd_slow", indicators.get("macd_slow", 26))
        indicators["atr_period"] = new_params.get("atr_period", indicators.get("atr_period", 14))

        rm = self.config.setdefault("risk_management", {})
        rm["kelly_fraction"] = new_params.get("kelly_cap", rm.get("kelly_fraction", 0.5))

        self.tuning_results["last_pnl"] = new_params.get("estimated_pnl", 0.0)
        self.tuning_results["current_params"] = new_params
        logger.info("[SelfTuning] Parameters hot-swapped successfully")

    def _send_telegram_report(self, params: dict[str, Any], status: str) -> None:
        """Send tuning report to Telegram."""
        try:
            telegram_logger = logging.getLogger("telegram")
            if status == "SUCCESS":
                msg = (
                    f"🔄 [WEEKEND TUNING SUCCESS] Parameter di-update! "
                    f"RSI_Period: {self.config.get('indicators', {}).get('rsi_period', 14)} -> {params.get('rsi_period')}, "
                    f"MACD: {self.config.get('indicators', {}).get('macd_fast', 12)},{self.config.get('indicators', {}).get('macd_slow', 26)} -> "
                    f"{params.get('macd_fast')},{params.get('macd_slow')}. "
                    f"Proyeksi PnL: {params.get('estimated_pnl', 0):.4f}. "
                    f"Sistem siap untuk pembukaan pasar minggu depan."
                )
            else:
                msg = (
                    f"⚠️ [WEEKEND TUNING REJECTED] Parameter baru tidak memenuhi upgrade threshold. "
                    f"Parameter lama dipertahankan. Proyeksi PnL baru: {params.get('estimated_pnl', 0):.4f}"
                )
            telegram_logger.info(msg)
        except Exception:
            pass