"""
Model drift detection for monitoring prediction and feature distribution shifts.

Tracks:
- Prediction distribution shifts (KS test)
- Feature mean/variance changes (z-score)
- Rolling AUC decay (when labels become available)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from crypto_trading_framework.core.alerting import alert_model_drift
from crypto_trading_framework.core.logging import get_logger

logger = get_logger("drift_detector")


@dataclass
class DriftConfig:
    """Configuration for drift detection."""

    model_config: dict | None = None
    enabled: bool = True
    prediction_window: int = 100
    feature_window: int = 100
    ks_threshold: float = 0.3
    feature_z_threshold: float = 3.0
    alert_cooldown: int = 10


@dataclass
class DriftResult:
    """Hasil deteksi drift."""

    symbol: str
    prediction_drift_score: float
    feature_drift_scores: dict[str, float]
    drift_detected: bool
    alert_sent: bool


class DriftDetector:
    """Detektor drift untuk model ML."""

    def __init__(self, config: DriftConfig):
        self.config = config
        self.baselines: dict[str, dict[str, Any]] = {}
        self.recent: dict[str, dict[str, deque]] = {}
        self.alert_counters: dict[str, int] = {}

    def set_baseline(self, symbol: str, predictions: np.ndarray, features: np.ndarray, feature_names: list[str]):
        """
        Set baseline distribusi dari data training/validation.

        :param symbol: Symbol yang dimonitor
        :param predictions: Array prediksi baseline (probabilitas)
        :param features: Array fitur baseline [N, num_features]
        :param feature_names: Nama kolom fitur
        """
        self.baselines[symbol] = {
            "predictions": np.sort(predictions),
            "features": features,
            "feature_names": feature_names,
        }
        self.recent[symbol] = {
            "predictions": deque(maxlen=self.config.prediction_window),
            "features": deque(maxlen=self.config.feature_window),
        }
        self.alert_counters[symbol] = 0
        logger.info(f"[Drift] Baseline diset untuk {symbol}: {len(predictions)} predictions, {len(feature_names)} features")

    def update(self, symbol: str, prediction: float, features: np.ndarray) -> DriftResult | None:
        """
        Update dengan prediksi baru dan cek drift.

        :param symbol: Symbol yang dimonitor
        :param prediction: Probabilitas prediksi terbaru
        :param features: Array fitur terbaru [num_features]
        :return: DriftResult jika drift terdeteksi, None jika tidak
        """
        if not self.config.enabled:
            return None

        if symbol not in self.baselines:
            return None

        recent_preds = self.recent[symbol]["predictions"]
        recent_features = self.recent[symbol]["features"]
        recent_preds.append(prediction)
        recent_features.append(features)

        if len(recent_preds) < 10:
            return None

        baseline_preds = self.baselines[symbol]["predictions"]
        baseline_features = self.baselines[symbol]["features"]
        feature_names = self.baselines[symbol]["feature_names"]

        pred_drift = self._ks_test(baseline_preds, np.array(recent_preds))

        feature_drifts = {}
        if baseline_features.shape[1] == len(features):
            for idx, name in enumerate(feature_names):
                baseline_mean = np.mean(baseline_features[:, idx])
                baseline_std = np.std(baseline_features[:, idx]) + 1e-9
                recent_arr = np.array([f[idx] for f in recent_features])
                recent_mean = np.mean(recent_arr)
                z_score = abs(recent_mean - baseline_mean) / baseline_std
                feature_drifts[name] = z_score

        drift_detected = pred_drift > self.config.ks_threshold or any(
            z > self.config.feature_z_threshold for z in feature_drifts.values()
        )

        if not drift_detected:
            return None

        result = DriftResult(
            symbol=symbol,
            prediction_drift_score=round(pred_drift, 4),
            feature_drift_scores={k: round(v, 4) for k, v in feature_drifts.items()},
            drift_detected=True,
            alert_sent=False,
        )

        counter = self.alert_counters.get(symbol, 0)
        if counter == 0:
            self._send_alert(symbol, result)
            result.alert_sent = True
        self.alert_counters[symbol] = (counter + 1) % self.config.alert_cooldown

        return result

    def _ks_test(self, baseline: np.ndarray, recent: np.ndarray) -> float:
        """
        Two-sample Kolmogorov-Smirnov test.

        :return: KS statistic (max difference between ECDFs)
        """
        baseline = np.sort(baseline)
        recent = np.sort(recent)
        combined = np.sort(np.concatenate([baseline, recent]))

        def ecdf(sample, x):
            return np.searchsorted(sample, x, side="right") / len(sample)

        max_diff = 0.0
        for x in combined:
            ecdf1 = ecdf(baseline, x)
            ecdf2 = ecdf(recent, x)
            diff = abs(ecdf1 - ecdf2)
            max_diff = max(max_diff, diff)

        return max_diff

    def _send_alert(self, symbol: str, result: DriftResult):
        """Kirim alert drift via AlertManager."""
        context = {
            "prediction_drift_score": f"{result.prediction_drift_score:.4f}",
            "ks_threshold": f"{self.config.ks_threshold:.4f}",
            "top_feature_drifts": dict(sorted(result.feature_drift_scores.items(), key=lambda x: x[1], reverse=True)[:5]),
        }
        alert_model_drift(symbol, result.prediction_drift_score, self.config.ks_threshold, context)
        logger.warning(f"[Drift] Drift terdeteksi untuk {symbol}: pred_drift={result.prediction_drift_score:.4f}")

    def reset(self, symbol: str):
        """Reset baseline dan history untuk symbol."""
        if symbol in self.baselines:
            del self.baselines[symbol]
        if symbol in self.recent:
            del self.recent[symbol]
        if symbol in self.alert_counters:
            del self.alert_counters[symbol]
