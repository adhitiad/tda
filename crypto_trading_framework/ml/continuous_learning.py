import hashlib
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import torch
from torch import nn

from crypto_trading_framework.core.logging import get_logger
from crypto_trading_framework.ml.ml_pipeline import MLPipeline
from crypto_trading_framework.ml.model import create_model

logger = get_logger("phase4.continuous_learning")

GOLDEN_MEMORY_FILE = "models/golden_memory.json"
BEST_WEIGHTS_FILE = "models/best_weights.pth"
SCALER_FILE = "models/scaler.pkl"
SIGNAL_HISTORY_FILE = "models/signal_history.jsonl"

POLARS_INDICATOR_COLS = [
    "close", "volume", "ema_20", "ema_50", "bb_width",
    "rsi", "stoch_k", "stoch_d", "atr", "macd_hist",
    "volume_ratio", "adx", "obv", "vwap",
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_timestamp(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S%z")


def _hash_sequence(values: dict[str, float]) -> str:
    canonical = json.dumps(values, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


class GoldenMemoryManager:
    def __init__(self, memory_path: str = GOLDEN_MEMORY_FILE, max_patterns: int = 5000, recency_decay_alpha: float = 0.95):
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_patterns = max_patterns
        self.recency_decay_alpha = recency_decay_alpha
        self._df: pl.DataFrame | None = None

    def _load_df(self) -> pl.DataFrame:
        if self._df is not None:
            return self._df
        if not self.memory_path.exists() or self.memory_path.stat().st_size == 0:
            self._df = pl.DataFrame({
                "sequence_hash": pl.Series([], dtype=pl.Utf8),
                "indicator_values": pl.Series([], dtype=pl.Utf8),
                "outcome": pl.Series([], dtype=pl.Utf8),
                "profit_pct": pl.Series([], dtype=pl.Float64),
                "direction": pl.Series([], dtype=pl.Utf8),
                "timestamp": pl.Series([], dtype=pl.Utf8),
                "recency_weight": pl.Series([], dtype=pl.Float64),
                "profitability_score": pl.Series([], dtype=pl.Float64),
            })
            return self._df
        with open(self.memory_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if not records:
            self._df = pl.DataFrame({
                "sequence_hash": pl.Series([], dtype=pl.Utf8),
                "indicator_values": pl.Series([], dtype=pl.Utf8),
                "outcome": pl.Series([], dtype=pl.Utf8),
                "profit_pct": pl.Series([], dtype=pl.Float64),
                "direction": pl.Series([], dtype=pl.Utf8),
                "timestamp": pl.Series([], dtype=pl.Utf8),
                "recency_weight": pl.Series([], dtype=pl.Float64),
                "profitability_score": pl.Series([], dtype=pl.Float64),
            })
            return self._df
        self._df = pl.DataFrame(records)
        return self._df

    def _save_df(self, df: pl.DataFrame) -> None:
        self._df = df
        records = df.to_dicts()
        with open(self.memory_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(rec, default=str) + "\n" for rec in records)

    def clear(self) -> None:
        self._df = None
        if self.memory_path.exists():
            self.memory_path.unlink()

    def add_pattern(
        self,
        indicator_values: dict[str, float],
        outcome: str,
        profit_pct: float,
        direction: str,
        sequence_hash: str | None = None,
    ) -> bool:
        hash_val = sequence_hash or _hash_sequence(indicator_values)
        df = self._load_df()
        existing = df.filter(pl.col("sequence_hash") == hash_val)
        if not existing.is_empty():
            return False

        now = _now_utc()
        recency_weight = 1.0
        profitability_score = self._compute_profitability_score(profit_pct, outcome)

        new_row = pl.DataFrame({
            "sequence_hash": [hash_val],
            "indicator_values": [json.dumps(indicator_values, sort_keys=True)],
            "outcome": [outcome],
            "profit_pct": [float(profit_pct)],
            "direction": [direction],
            "timestamp": [_serialize_timestamp(now)],
            "recency_weight": [recency_weight],
            "profitability_score": [profitability_score],
        })
        df = pl.concat([df, new_row], how="diagonal")
        df = self._apply_recency_decay(df, alpha=self.recency_decay_alpha)
        df = self._resolve_conflicts(df, hash_val)
        df = self._trim_max_patterns(df)
        self._save_df(df)
        return True

    def add_incorrect_pattern(self, indicator_values: dict[str, float], sequence_hash: str | None = None) -> None:
        hash_val = sequence_hash or _hash_sequence(indicator_values)
        df = self._load_df()
        df = df.filter(pl.col("sequence_hash") != hash_val)
        self._save_df(df)

    def get_patterns_by_outcome(self, outcome: str) -> pl.DataFrame:
        df = self._load_df()
        return df.filter(pl.col("outcome") == outcome)

    def get_all_patterns(self) -> pl.DataFrame:
        return self._load_df()

    def get_correct_pattern_count(self) -> int:
        df = self._load_df()
        return df.filter(pl.col("outcome") == "correct").height

    def get_incorrect_pattern_count(self) -> int:
        df = self._load_df()
        return df.filter(pl.col("outcome") == "incorrect").height

    def _compute_profitability_score(self, profit_pct: float, outcome: str) -> float:
        if outcome == "correct":
            return min(1.0, max(0.0, profit_pct / 100.0 + 0.5))
        return max(0.0, min(1.0, abs(profit_pct) / 100.0))

    def _apply_recency_decay(self, df: pl.DataFrame, alpha: float = 0.95) -> pl.DataFrame:
        if df.is_empty():
            return df
        now = _now_utc()
        now_epoch_us = int(now.timestamp() * 1_000_000)
        parsed = pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%z", strict=False)
        age_days = (pl.lit(now_epoch_us) - parsed.dt.epoch()).cast(pl.Float64) / 86400.0 / 1_000_000.0
        return df.with_columns(age_days.pow(pl.lit(alpha)).alias("recency_weight"))

    def _resolve_conflicts(self, df: pl.DataFrame, new_hash: str | None = None) -> pl.DataFrame:
        if df.is_empty():
            return df
        if new_hash is None:
            return df

        existing = df.filter(pl.col("sequence_hash") == new_hash)
        if existing.height <= 1:
            return df

        df = df.with_columns(
            (pl.col("recency_weight") * pl.col("profitability_score")).alias("combined_score")
        )
        best_idx = df.select(pl.col("combined_score").arg_max()).item()
        if best_idx is None:
            return df

        kept_row = df.slice(best_idx, 1)
        other_hashes = df.filter(pl.col("sequence_hash") != new_hash)
        result = pl.concat([other_hashes, kept_row], how="diagonal")
        result = result.drop("combined_score")
        return result

    def _trim_max_patterns(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height <= self.max_patterns:
            return df
        df = df.with_columns(
            (pl.col("recency_weight") * pl.col("profitability_score")).alias("combined_score")
        )
        df = df.sort("combined_score", descending=True).slice(0, self.max_patterns)
        df = df.drop("combined_score")
        return df

    def get_golden_dataset(self) -> pl.DataFrame:
        df = self.get_patterns_by_outcome("correct")
        if df.is_empty():
            return df
        return df.with_columns(
            pl.col("indicator_values").map_elements(json.loads).alias("indicators_parsed")
        )

    def get_metadata(self) -> dict[str, Any]:
        df = self._load_df()
        correct = df.filter(pl.col("outcome") == "correct").height
        incorrect = df.filter(pl.col("outcome") == "incorrect").height
        return {
            "total_patterns": df.height,
            "correct_count": correct,
            "incorrect_count": incorrect,
            "last_updated": _serialize_timestamp(_now_utc()),
            "version": "1.0",
        }


class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.best_weights_path = self.models_dir / "best_weights.pth"
        self.scaler_path = self.models_dir / "scaler.pkl"

    def save_best_model(self, model: nn.Module, scaler: object, current_loss: float) -> bool:
        previous_loss = self._load_previous_loss()
        if previous_loss is not None and current_loss >= previous_loss:
            logger.info(
                f"[ModelManager] New loss ({current_loss:.6f}) >= previous loss ({previous_loss:.6f}). "
                f"Keeping old weights."
            )
            return False

        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), str(self.best_weights_path))
        else:
            joblib.dump(model, str(self.best_weights_path))

        joblib.dump(scaler, str(self.scaler_path))
        self._save_loss_marker(current_loss)
        logger.info(f"[ModelManager] Model overwritten with loss={current_loss:.6f}")
        return True

    def load_model(self, model_type: str = "lstm", input_size: int = 11, device: torch.device | str = "cpu") -> nn.Module | None:
        if not self.best_weights_path.exists():
            logger.warning("[ModelManager] No best_weights.pth found. Cannot load model.")
            return None
        model = create_model(
            model_type=model_type,
            input_size=input_size,
        ).to(device)
        state_dict = torch.load(str(self.best_weights_path), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"[ModelManager] Loaded model from {self.best_weights_path}")
        return model

    def load_scaler(self) -> object | None:
        if not self.scaler_path.exists():
            logger.warning("[ModelManager] No scaler.pkl found.")
            return None
        return joblib.load(str(self.scaler_path))

    def model_exists(self) -> bool:
        return self.best_weights_path.exists() and self.scaler_path.exists()

    def _load_previous_loss(self) -> float | None:
        marker_path = self.models_dir / ".best_loss.marker"
        if not marker_path.exists():
            return None
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                return float(f.read().strip())
        except (ValueError, OSError):
            return None

    def _save_loss_marker(self, loss: float) -> None:
        marker_path = self.models_dir / ".best_loss.marker"
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(str(loss))


class SignalBuffer:
    def __init__(self, history_path: str = SIGNAL_HISTORY_FILE):
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def store_signal(
        self,
        indicator_values: dict[str, float],
        direction: str,
        entry_price: float,
        probability: float,
        tp_pct: float = 0.03,
        sl_pct: float = 0.015,
        timeframe: str = "m15",
        symbol: str = "BTC/USDT:USDT",
    ) -> None:
        record = {
            "indicator_values": indicator_values,
            "direction": direction,
            "entry_price": entry_price,
            "probability": probability,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "timeframe": timeframe,
            "symbol": symbol,
            "timestamp": _serialize_timestamp(_now_utc()),
            "evaluated": False,
            "outcome": "",
            "profit_pct": 0.0,
        }
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def get_unevaluated_signals(self, max_age_hours: float = 6.0) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        records = []
        with open(self.history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("evaluated", False):
                    continue
                try:
                    ts = datetime.fromisoformat(rec["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age = (_now_utc() - ts).total_seconds() / 3600.0
                    if age < max_age_hours:
                        rec["_age_hours"] = age
                        records.append(rec)
                except (ValueError, KeyError):
                    continue
        return records

    def mark_evaluated(self, signal: dict[str, Any], outcome: str, profit_pct: float) -> None:
        signal["evaluated"] = True
        signal["outcome"] = outcome
        signal["profit_pct"] = profit_pct

    def mark_evaluated_by_index(self, index: int, outcome: str, profit_pct: float) -> None:
        records = self._load_all_records()
        if 0 <= index < len(records):
            records[index]["evaluated"] = True
            records[index]["outcome"] = outcome
            records[index]["profit_pct"] = profit_pct
            self._write_all_records(records)

    def _load_all_records(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        records = []
        with open(self.history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _write_all_records(self, records: list[dict[str, Any]]) -> None:
        with open(self.history_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(rec, default=str) + "\n" for rec in records)

    def clear_evaluated(self) -> None:
        records = self._load_all_records()
        unevaluated = [r for r in records if not r.get("evaluated", False)]
        self._write_all_records(unevaluated)
        logger.info(f"[SignalBuffer] Cleared evaluated signals. Kept {len(unevaluated)} unevaluated.")

    def clear_all(self) -> None:
        if self.history_path.exists():
            self.history_path.unlink()


def _fetch_actual_price_after(
    df_history: pl.DataFrame,
    signal_timestamp: str,
    lookahead_candles: int = 20,
) -> dict[str, Any] | None:
    try:
        ts = datetime.fromisoformat(signal_timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None

    # Check if timestamp column is already datetime or needs parsing
    ts_dtype = df_history.schema.get("timestamp")
    if ts_dtype and ts_dtype.is_(pl.Utf8):
        df = df_history.with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%z", strict=False)
        ).sort("timestamp")
    else:
        df = df_history.sort("timestamp")

    # Make comparison timestamp timezone-naive to match Polars datetime (which is tz-naive)
    ts_naive = ts.replace(tzinfo=None)
    mask = df["timestamp"] > ts_naive
    future_df = df.filter(mask)
    if future_df.height == 0:
        return None

    future_df = future_df.limit(lookahead_candles)
    entry_price = future_df["open"].head(1).item()
    highest = future_df["high"].max()
    lowest = future_df["low"].min()
    close = future_df["close"].tail(1).item()

    return {
        "entry_price": float(entry_price) if entry_price is not None else None,
        "high": float(highest) if highest is not None else None,
        "low": float(lowest) if lowest is not None else None,
        "close": float(close) if close is not None else None,
    }


def _evaluate_signal_profit(
    signal: dict[str, Any],
    actual_data: dict[str, Any],
) -> tuple[str, float]:
    if actual_data is None or actual_data.get("entry_price") is None:
        return "uncertain", 0.0

    direction = signal.get("direction", "LONG")
    entry_price = actual_data["entry_price"]
    tp_pct = signal.get("tp_pct", 0.03)
    sl_pct = signal.get("sl_pct", 0.015)

    tp_price = entry_price * (1 + tp_pct) if direction == "LONG" else entry_price * (1 - tp_pct)
    sl_price = entry_price * (1 - sl_pct) if direction == "LONG" else entry_price * (1 + sl_pct)

    high = actual_data.get("high", entry_price)
    low = actual_data.get("low", entry_price)
    close = actual_data.get("close", entry_price)

    hit_tp = direction == "LONG" and high >= tp_price
    hit_tp = hit_tp or (direction == "SHORT" and low <= tp_price)
    hit_sl = direction == "LONG" and low <= sl_price
    hit_sl = hit_sl or (direction == "SHORT" and high >= sl_price)

    if hit_tp:
        profit_pct = tp_pct * 100
        return "correct", profit_pct
    if hit_sl:
        profit_pct = -sl_pct * 100
        return "incorrect", profit_pct

    direction_sign = 1 if direction == "LONG" else -1
    price_change = (close - entry_price) / entry_price * 100 * direction_sign
    if price_change > 0:
        return "correct", price_change
    else:
        return "incorrect", price_change


class FeedbackLoopEngine:
    def __init__(self, config: dict | None = None):
        from crypto_trading_framework.core.config_schema import (
            validate_config,
        )
        if config is None:
            config = {}
        validated = validate_config(config)
        self.config = validated

        self.enabled = validated.get("continuous_learning", {}).get("enabled", True)
        self.evaluation_interval_hours = validated.get("continuous_learning", {}).get(
            "evaluation_interval_hours", 2.0
        )
        self.retrain_interval_hours = validated.get("continuous_learning", {}).get(
            "retrain_interval_hours", 24.0
        )
        self.lookback_hours = validated.get("continuous_learning", {}).get(
            "lookback_hours", 6.0
        )
        self.min_golden_samples = validated.get("continuous_learning", {}).get(
            "min_golden_samples", 50
        )
        self.model_type = validated.get("model", {}).get("type", "lstm")
        self.time_steps = validated.get("ml", {}).get("time_steps", 60)
        self.feature_cols = validated.get("ml", {}).get("feature_cols", [])
        self.input_size = max(len(self.feature_cols), 11)

        self.golden_memory = GoldenMemoryManager(
            max_patterns=validated.get("continuous_learning", {}).get("max_memory_patterns", 5000),
            recency_decay_alpha=validated.get("continuous_learning", {}).get("recency_decay_alpha", 0.95),
        )
        self.model_manager = ModelManager()
        self.signal_buffer = SignalBuffer()

        self._last_evaluation: datetime | None = None
        self._last_retrain: datetime | None = None
        self._running = False
        self._background_thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            logger.info("[FeedbackLoop] Continuous learning is disabled in config.")
            return
        if self._running:
            logger.warning("[FeedbackLoop] Feedback loop already running.")
            return
        self._running = True
        self._background_thread = threading.Thread(
            target=self._background_loop, name="continuous-learning-loop", daemon=True
        )
        self._background_thread.start()
        logger.info("[FeedbackLoop] Continuous Learning Engine started.")

    def stop(self) -> None:
        self._running = False
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=10)
        logger.info("[FeedbackLoop] Continuous Learning Engine stopped.")

    def is_running(self) -> bool:
        return self._running

    def _background_loop(self) -> None:
        while self._running:
            try:
                now = _now_utc()
                self._run_evaluation(now)
                self._run_retrain(now)
                self._cleanup_signal_buffer()
            except Exception:
                logger.exception("[FeedbackLoop] Error in background loop")
            time.sleep(300)

    def _run_evaluation(self, now: datetime) -> None:
        evaluation_interval = timedelta(hours=self.evaluation_interval_hours)
        if self._last_evaluation is not None and (now - self._last_evaluation) < evaluation_interval:
            return

        logger.info("[FeedbackLoop] Running signal evaluation cycle...")
        self._last_evaluation = now

        unevaluated = self.signal_buffer.get_unevaluated_signals(max_age_hours=self.lookback_hours)
        if not unevaluated:
            logger.info("[FeedbackLoop] No unevaluated signals found.")
            return

        logger.info(f"[FeedbackLoop] Evaluating {len(unevaluated)} unevaluated signal(s)...")
        for sig in unevaluated:
            try:
                self._evaluate_single_signal(sig)
            except Exception:
                logger.exception("[FeedbackLoop] Error evaluating signal")

        self.signal_buffer.clear_evaluated()
        logger.info(
            f"[FeedbackLoop] Evaluation complete. "
            f"Golden memory: {self.golden_memory.get_correct_pattern_count()} correct, "
            f"{self.golden_memory.get_incorrect_pattern_count()} incorrect patterns."
        )

    def _evaluate_single_signal(self, signal: dict[str, Any]) -> None:
        indicator_values = signal.get("indicator_values", {})
        if not indicator_values:
            self.signal_buffer.mark_evaluated(signal, "uncertain", 0.0)
            return

        try:
            ts = datetime.fromisoformat(signal["timestamp"])
        except (ValueError, KeyError):
            self.signal_buffer.mark_evaluated(signal, "uncertain", 0.0)
            return

        actual_data = self._fetch_historical_prices(signal, ts)
        outcome, profit_pct = _evaluate_signal_profit(signal, actual_data)

        sequence_hash = _hash_sequence(indicator_values)

        if outcome == "correct":
            self.golden_memory.add_pattern(
                indicator_values=indicator_values,
                outcome="correct",
                profit_pct=profit_pct,
                direction=signal.get("direction", "LONG"),
                sequence_hash=sequence_hash,
            )
        elif outcome == "incorrect":
            self.golden_memory.add_incorrect_pattern(indicator_values, sequence_hash=sequence_hash)

        self.signal_buffer.mark_evaluated(signal, outcome, profit_pct)
        logger.info(
            f"[FeedbackLoop] Signal evaluated: outcome={outcome}, profit={profit_pct:+.2f}%, "
            f"direction={signal.get('direction', 'UNKNOWN')}"
        )

    def _fetch_historical_prices(
        self, signal: dict[str, Any], signal_timestamp: datetime
    ) -> dict[str, Any] | None:
        from crypto_trading_framework.data_ingestion import DataIngestion

        symbol = signal.get("symbol", "BTC/USDT:USDT")
        timeframe = signal.get("timeframe", "m15")
        interval_map = {"m5": "5m", "m15": "15m", "m30": "30m", "h1": "1h", "h4": "4h", "d1": "1d"}
        interval = interval_map.get(timeframe, "1h")

        try:
            import asyncio
            ingestion = DataIngestion(
                exchange_id=self.config.get("data", {}).get("exchange_id", "okx"),
                symbol=symbol,
            )
            raw_data_dict = asyncio.run(ingestion.fetch_multi_timeframe(timeframes=[timeframe], limit=500))
            if raw_data_dict and timeframe in raw_data_dict:
                raw_data = raw_data_dict[timeframe]
            else:
                raw_data = None
            asyncio.run(ingestion.close())
            if raw_data is None or raw_data.is_empty():
                return None
            return _fetch_actual_price_after(raw_data, signal_timestamp.isoformat())
        except Exception:
            logger.exception("[FeedbackLoop] Could not fetch historical prices")
            return None

    def _run_retrain(self, now: datetime) -> None:
        retrain_interval = timedelta(hours=self.retrain_interval_hours)
        if self._last_retrain is not None and (now - self._last_retrain) < retrain_interval:
            return

        correct_count = self.golden_memory.get_correct_pattern_count()
        if correct_count < self.min_golden_samples:
            logger.info(
                f"[FeedbackLoop] Skipping retrain: only {correct_count} golden patterns "
                f"(minimum {self.min_golden_samples} required)."
            )
            return

        logger.info("[FeedbackLoop] Starting automatic model retraining...")
        self._last_retrain = now

        try:
            self.auto_retrain_model()
        except Exception:
            logger.exception("[FeedbackLoop] Retraining failed")

    def run_evaluation_cycle(self) -> dict[str, Any]:
        now = _now_utc()
        self._run_evaluation(now)
        return {
            "evaluated_at": _serialize_timestamp(now),
            "golden_correct": self.golden_memory.get_correct_pattern_count(),
            "golden_incorrect": self.golden_memory.get_incorrect_pattern_count(),
        }

    def auto_retrain_model(self) -> dict[str, Any]:
        golden_df = self.golden_memory.get_golden_dataset()
        if golden_df.is_empty():
            logger.warning("[FeedbackLoop] No golden patterns available for retraining.")
            return {"status": "skipped", "reason": "no_golden_data"}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = MLPipeline(scaler_type=self.config.get("ml", {}).get("scaler_type", "minmax"))

        indicator_rows = golden_df.select("indicators_parsed").to_numpy().tolist()
        flat_indicators = []
        for row in indicator_rows:
            if isinstance(row, list) and len(row) > 0:
                flat_indicators.append(row[0])
            else:
                flat_indicators.append({})

        if not flat_indicators:
            return {"status": "skipped", "reason": "no_indicator_data"}

        all_keys = sorted({k for d in flat_indicators for k in d})
        indicator_matrix = np.array(
            [[d.get(k, 0.0) for k in all_keys] for d in flat_indicators],
            dtype=np.float32,
        )

        directions = golden_df.select("direction").to_numpy().flatten()
        labels = np.where(directions == "LONG", 1, 0).astype(np.float32)

        if len(indicator_matrix) < self.time_steps + 1:
            logger.warning(
                f"[FeedbackLoop] Not enough golden samples ({len(indicator_matrix)}) "
                f"for sequence length {self.time_steps}."
            )
            return {"status": "skipped", "reason": "insufficient_samples", "samples": len(indicator_matrix)}

        scaled = pipeline.scale_features(indicator_matrix, fit=True)
        X_seq, y_seq = pipeline.create_sequences(scaled, labels, time_steps=self.time_steps)

        if len(X_seq) == 0:
            return {"status": "skipped", "reason": "no_sequences_created"}

        split_idx = max(1, int(len(X_seq) * 0.8))
        X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
        X_val, y_val = X_seq[split_idx:], y_seq[split_idx:]

        if len(X_train) == 0 or len(X_val) == 0:
            return {"status": "skipped", "reason": "train_val_split_failed"}

        input_size = X_train.shape[2]
        model = create_model(
            model_type=self.model_type,
            input_size=input_size,
        ).to(device)

        previous_weights_path = self.model_manager.best_weights_path
        if previous_weights_path.exists():
            try:
                state_dict = torch.load(str(previous_weights_path), map_location=device, weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                logger.info("[FeedbackLoop] Loaded previous weights for incremental fine-tuning.")
            except Exception:
                logger.warning("[FeedbackLoop] Could not load previous weights for incremental learning")

        scaler = pipeline.scaler

        num_pos = float((y_train == 1).sum())
        num_neg = float((y_train == 0).sum())
        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        x_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

        best_val_loss = float("inf")
        best_model_state: dict[str, torch.Tensor] | None = None
        patience_counter = 0

        for epoch in range(30):
            optimizer.zero_grad()
            output = model(x_train_t)
            loss = loss_fn(output, y_train_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    x_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
                    val_output = model(x_val_t)
                    val_loss = loss_fn(val_output, y_val_t).item()
                model.train()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        with torch.no_grad():
            x_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
            val_logits = model(x_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
            val_preds = (val_probs >= 0.5).astype(int)
            val_accuracy = float(np.mean(val_preds == y_val.cpu().numpy()))

        if self.model_manager.save_best_model(model, scaler, best_val_loss):
            logger.info(f"[FeedbackLoop] Retrained model saved. Val accuracy={val_accuracy:.4f}, Val loss={best_val_loss:.6f}")
        else:
            logger.info(
                f"[FeedbackLoop] Retrained model evaluated (val acc={val_accuracy:.4f}, "
                f"val loss={best_val_loss:.6f}) but NOT saved (not better than current)."
            )

        return {
            "status": "completed",
            "val_accuracy": val_accuracy,
            "val_loss": best_val_loss,
            "golden_samples": len(X_seq),
            "train_samples": split_idx,
            "val_samples": len(X_seq) - split_idx,
        }

    def _cleanup_signal_buffer(self) -> None:
        try:
            self.signal_buffer.clear_evaluated()
        except Exception:
            logger.exception("[FeedbackLoop] Error cleaning up signal buffer")

    def store_signal(
        self,
        indicator_values: dict[str, float],
        direction: str,
        entry_price: float,
        probability: float,
        tp_pct: float = 0.03,
        sl_pct: float = 0.015,
        timeframe: str = "m15",
        symbol: str = "BTC/USDT:USDT",
    ) -> None:
        self.signal_buffer.store_signal(
            indicator_values=indicator_values,
            direction=direction,
            entry_price=entry_price,
            probability=probability,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            timeframe=timeframe,
            symbol=symbol,
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "enabled": self.enabled,
            "golden_correct_patterns": self.golden_memory.get_correct_pattern_count(),
            "golden_incorrect_patterns": self.golden_memory.get_incorrect_pattern_count(),
            "model_exists": self.model_manager.model_exists(),
            "last_evaluation": _serialize_timestamp(self._last_evaluation) if self._last_evaluation else None,
            "last_retrain": _serialize_timestamp(self._last_retrain) if self._last_retrain else None,
            "evaluation_interval_hours": self.evaluation_interval_hours,
            "retrain_interval_hours": self.retrain_interval_hours,
        }