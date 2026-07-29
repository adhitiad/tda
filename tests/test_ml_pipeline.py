import numpy as np
import polars as pl
import pytest

from crypto_trading_framework.ml.ml_pipeline import MLPipeline


@pytest.fixture
def sample_features():
    np.random.seed(42)
    n = 500
    return np.random.randn(n, 11).astype(np.float32)


@pytest.fixture
def sample_targets(sample_features):
    return np.random.randint(0, 2, len(sample_features)).astype(np.float32)


def test_prepare_features():
    df = pl.DataFrame({
        "close": np.random.randn(100),
        "volume": np.random.randn(100),
        "ema_20": np.random.randn(100),
        "ema_50": np.random.randn(100),
        "rsi": np.random.randn(100),
    })
    pipeline = MLPipeline()
    features, cols = pipeline.prepare_features(df)
    assert features.shape[1] == 5
    assert len(cols) == 5


def test_scale_features(sample_features):
    pipeline = MLPipeline(scaler_type="minmax")
    scaled = pipeline.scale_features(sample_features, fit=True)
    assert scaled.min() >= 0
    assert scaled.max() <= 1 + 1e-5


def test_define_target():
    df = pl.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    pipeline = MLPipeline()
    df = pipeline.define_target(df, forward_periods=2)
    assert "target" in df.columns
    assert df["target"][0] == 1  # 3 > 1
    assert df["target"][1] == 1  # 4 > 2


def test_create_sequences(sample_features, sample_targets):
    pipeline = MLPipeline()
    X, y = pipeline.create_sequences(sample_features, sample_targets, time_steps=10)
    assert X.shape[0] == len(sample_features) - 10
    assert X.shape[1] == 10
    assert X.shape[2] == sample_features.shape[1]
    assert len(y) == len(sample_features) - 10


def test_train_test_split_sequences(sample_features, sample_targets):
    pipeline = MLPipeline()
    X, y = pipeline.create_sequences(sample_features, sample_targets, time_steps=10)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    assert len(X_train) + len(X_test) == len(X)


def test_save_and_load_scaler(tmp_path):
    pipeline = MLPipeline(scaler_type="minmax")
    features = np.random.randn(100, 5).astype(np.float32)
    pipeline.scale_features(features, fit=True)

    scaler_path = str(tmp_path / "scaler.pkl")
    pipeline.save_scaler(scaler_path)

    new_pipeline = MLPipeline()
    new_pipeline.load_scaler(scaler_path)
    scaled = new_pipeline.scale_features(features, fit=False)
    assert np.allclose(scaled.min(), 0, atol=1e-5)
    assert np.allclose(scaled.max(), 1, atol=1e-5)


def test_walk_forward_split(sample_features, sample_targets):
    pipeline = MLPipeline()
    X, y = pipeline.create_sequences(sample_features, sample_targets, time_steps=10)
    splits = pipeline.walk_forward_split(X, y, n_splits=3)
    assert len(splits) == 2
    for X_train, X_val, y_train, y_val in splits:
        assert len(X_train) > 0
        assert len(X_val) > 0
