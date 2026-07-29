"""
Tests for model drift detection.
"""

import numpy as np

from crypto_trading_framework.ml.drift_detector import (
    DriftConfig,
    DriftDetector,
    DriftResult,
)


def _make_baseline(n=200):
    np.random.seed(42)
    preds = np.random.beta(2, 5, n)
    features = np.random.randn(n, 4)
    return preds, features, ["f1", "f2", "f3", "f4"]


class TestDriftConfig:
    def test_defaults(self):
        cfg = DriftConfig()
        assert cfg.enabled is True
        assert cfg.prediction_window == 100
        assert cfg.ks_threshold == 0.3
        assert cfg.feature_z_threshold == 3.0
        assert cfg.alert_cooldown == 10

    def test_custom(self):
        cfg = DriftConfig(enabled=False, ks_threshold=0.5, feature_z_threshold=2.0)
        assert cfg.enabled is False
        assert cfg.ks_threshold == 0.5
        assert cfg.feature_z_threshold == 2.0


class TestDriftDetectorBaseline:
    def test_set_baseline(self):
        detector = DriftDetector(DriftConfig())
        preds, feats, names = _make_baseline()
        detector.set_baseline("BTC/USDT", preds, feats, names)
        assert "BTC/USDT" in detector.baselines
        assert "BTC/USDT" in detector.recent

    def test_set_baseline_multiple_symbols(self):
        detector = DriftDetector(DriftConfig())
        preds, feats, names = _make_baseline()
        detector.set_baseline("BTC/USDT", preds, feats, names)
        detector.set_baseline("ETH/USDT", preds, feats, names)
        assert len(detector.baselines) == 2
        assert len(detector.recent) == 2


class TestDriftDetectorUpdate:
    def test_no_drift_returns_none(self):
        detector = DriftDetector(DriftConfig(ks_threshold=0.8, feature_z_threshold=10.0))
        preds, feats, names = _make_baseline()
        detector.set_baseline("BTC/USDT", preds, feats, names)
        rng = np.random.default_rng(42)
        for _ in range(50):
            p = float(rng.beta(2, 5, 1)[0])
            result = detector.update("BTC/USDT", p, rng.standard_normal(4))
        assert result is None

    def test_drift_detected(self):
        detector = DriftDetector(DriftConfig(ks_threshold=0.1, feature_z_threshold=2.0))
        preds, feats, names = _make_baseline()
        detector.set_baseline("BTC/USDT", preds, feats, names)
        for _ in range(20):
            detector.update("BTC/USDT", 0.99, np.array([10.0, 0.0, 0.0, 0.0]))
        result = detector.update("BTC/USDT", 0.99, np.array([10.0, 0.0, 0.0, 0.0]))
        assert result is not None
        assert bool(result.drift_detected) is True

    def test_disabled_returns_none(self):
        detector = DriftDetector(DriftConfig(enabled=False))
        preds, feats, names = _make_baseline()
        detector.set_baseline("BTC/USDT", preds, feats, names)
        result = detector.update("BTC/USDT", 0.99, np.array([10.0, 0.0, 0.0, 0.0]))
        assert result is None

    def test_unknown_symbol_returns_none(self):
        detector = DriftDetector(DriftConfig())
        result = detector.update("UNKNOWN", 0.5, np.random.randn(4))
        assert result is None

    def test_insufficient_history_returns_none(self):
        detector = DriftDetector(DriftConfig())
        preds, feats, names = _make_baseline()
        detector.set_baseline("BTC/USDT", preds, feats, names)
        result = detector.update("BTC/USDT", 0.5, np.random.randn(4))
        assert result is None

    def test_result_fields(self):
        detector = DriftDetector(DriftConfig(ks_threshold=0.1))
        preds, feats, names = _make_baseline()
        detector.set_baseline("BTC/USDT", preds, feats, names)
        for _ in range(20):
            detector.update("BTC/USDT", 0.99, np.array([10.0, 0.0, 0.0, 0.0]))
        result = detector.update("BTC/USDT", 0.99, np.array([10.0, 0.0, 0.0, 0.0]))
        assert isinstance(result, DriftResult)
        assert result.symbol == "BTC/USDT"
        assert isinstance(result.prediction_drift_score, float)
        assert isinstance(result.feature_drift_scores, dict)
        assert bool(result.drift_detected) is True

    def test_alert_cooldown(self):
        detector = DriftDetector(DriftConfig(ks_threshold=0.1, feature_z_threshold=2.0, alert_cooldown=2))
        preds, feats, names = _make_baseline()
        detector.set_baseline("BTC/USDT", preds, feats, names)
        for _ in range(20):
            detector.update("BTC/USDT", 0.99, np.array([10.0, 0.0, 0.0, 0.0]))
        detector.alert_counters["BTC/USDT"] = 0
        r1 = detector.update("BTC/USDT", 0.99, np.array([10.0, 0.0, 0.0, 0.0]))
        assert r1 is not None and bool(r1.alert_sent) is True
        r2 = detector.update("BTC/USDT", 0.99, np.array([10.0, 0.0, 0.0, 0.0]))
        assert r2 is not None and bool(r2.alert_sent) is False
        r3 = detector.update("BTC/USDT", 0.99, np.array([10.0, 0.0, 0.0, 0.0]))
        assert r3 is not None and bool(r3.alert_sent) is True

    def test_reset(self):
        detector = DriftDetector(DriftConfig())
        preds, feats, names = _make_baseline()
        detector.set_baseline("BTC/USDT", preds, feats, names)
        detector.reset("BTC/USDT")
        assert "BTC/USDT" not in detector.baselines
        assert "BTC/USDT" not in detector.recent


class TestKSTest:
    def test_identical_distributions(self):
        detector = DriftDetector(DriftConfig())
        baseline = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        recent = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        score = detector._ks_test(baseline, recent)
        assert score == 0.0

    def test_different_distributions(self):
        detector = DriftDetector(DriftConfig())
        baseline = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        recent = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
        score = detector._ks_test(baseline, recent)
        assert score > 0.5

    def test_partial_overlap(self):
        detector = DriftDetector(DriftConfig())
        baseline = np.random.beta(2, 5, 100)
        recent = np.random.beta(2, 5, 100)
        score = detector._ks_test(baseline, recent)
        assert 0.0 <= score <= 1.0
