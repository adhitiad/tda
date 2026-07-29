"""
Tests for data validation module.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from crypto_trading_framework.core.validation import DataValidator, ValidationResult, create_validator_from_config


def _make_df(rows=5, inject_errors=False):
    now = datetime(2024, 1, 1, 12, 0, 0)
    data = {
        "timestamp": [now + timedelta(hours=i) for i in range(rows)],
        "open": [100.0 + i for i in range(rows)],
        "high": [101.0 + i for i in range(rows)],
        "low": [99.0 + i for i in range(rows)],
        "close": [100.5 + i for i in range(rows)],
        "volume": [1000.0 + i * 10 for i in range(rows)],
    }
    if inject_errors:
        data["high"][0] = 50.0
        data["low"][1] = 200.0
        data["close"][2] = 200.0
        data["volume"][3] = -10.0
    return pl.DataFrame(data)


class TestValidationResult:
    def test_default_is_valid(self):
        result = ValidationResult()
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error_marks_invalid(self):
        result = ValidationResult()
        result.add_error("error")
        assert result.valid is False
        assert result.errors == ["error"]

    def test_add_warning_keeps_valid(self):
        result = ValidationResult()
        result.add_warning("warning")
        assert result.valid is True
        assert result.warnings == ["warning"]

    def test_repr(self):
        result = ValidationResult()
        result.add_error("e1")
        result.add_warning("w1")
        r = repr(result)
        assert "FAIL" in r
        assert "e1" in r
        assert "w1" in r


class TestDataValidator:
    def test_valid_df_passes(self):
        validator = DataValidator()
        df = _make_df()
        result = validator.validate(df)
        assert result.valid is True

    def test_missing_columns_fails(self):
        validator = DataValidator()
        df = _make_df().drop("volume")
        result = validator.validate(df)
        assert result.valid is False
        assert any("Kolom yang hilang" in e for e in result.errors)

    def test_null_values_fail(self):
        validator = DataValidator()
        df = _make_df()
        close_vals = df["close"].to_list()
        close_vals[0] = None
        df = df.with_columns(pl.Series("close", close_vals))
        result = validator.validate(df)
        assert result.valid is False
        assert any("null" in e.lower() for e in result.errors)

    def test_high_less_than_low_fails(self):
        validator = DataValidator()
        df = _make_df(inject_errors=True)
        result = validator.validate(df)
        assert result.valid is False
        assert any("high < low" in e for e in result.errors)

    def test_open_greater_than_high_fails(self):
        validator = DataValidator()
        df = _make_df()
        open_vals = df["open"].to_list()
        open_vals[0] = 200.0
        df = df.with_columns(pl.Series("open", open_vals))
        result = validator.validate(df)
        assert result.valid is False
        assert any("open > high" in e for e in result.errors)

    def test_close_greater_than_high_fails(self):
        validator = DataValidator()
        df = _make_df()
        close_vals = df["close"].to_list()
        close_vals[0] = 200.0
        df = df.with_columns(pl.Series("close", close_vals))
        result = validator.validate(df)
        assert result.valid is False
        assert any("close > high" in e for e in result.errors)

    def test_open_less_than_low_fails(self):
        validator = DataValidator()
        df = _make_df()
        open_vals = df["open"].to_list()
        open_vals[0] = 50.0
        df = df.with_columns(pl.Series("open", open_vals))
        result = validator.validate(df)
        assert result.valid is False
        assert any("open < low" in e for e in result.errors)

    def test_close_less_than_low_fails(self):
        validator = DataValidator()
        df = _make_df()
        close_vals = df["close"].to_list()
        close_vals[0] = 50.0
        df = df.with_columns(pl.Series("close", close_vals))
        result = validator.validate(df)
        assert result.valid is False
        assert any("close < low" in e for e in result.errors)

    def test_negative_volume_fails(self):
        validator = DataValidator()
        df = _make_df()
        vol_vals = df["volume"].to_list()
        vol_vals[0] = -5.0
        df = df.with_columns(pl.Series("volume", vol_vals))
        result = validator.validate(df)
        assert result.valid is False
        assert any("volume < 0" in e for e in result.errors)

    def test_outlier_warning(self):
        validator = DataValidator(max_price_change_pct=0.01)
        df = _make_df()
        close_vals = df["close"].to_list()
        close_vals[1] = df["high"][1] * 0.99
        df = df.with_columns(pl.Series("close", close_vals))
        result = validator.validate(df)
        assert result.valid is True
        assert any("outlier" in w.lower() for w in result.warnings)

    def test_gap_warning(self):
        validator = DataValidator(allow_gaps=False)
        df = _make_df(rows=10)
        ts_vals = df["timestamp"].to_list()
        ts_vals[5] = ts_vals[5] + timedelta(hours=10)
        df = df.with_columns(pl.Series("timestamp", ts_vals))
        result = validator.validate(df)
        assert any("gap" in w.lower() for w in result.warnings)

    def test_allow_gaps_skips_gap_check(self):
        validator = DataValidator(allow_gaps=True)
        df = _make_df(rows=10)
        ts_vals = df["timestamp"].to_list()
        ts_vals[5] = ts_vals[5] + timedelta(hours=10)
        df = df.with_columns(pl.Series("timestamp", ts_vals))
        result = validator.validate(df)
        assert not any("gap" in w.lower() for w in result.warnings)

    def test_empty_df_fails(self):
        validator = DataValidator()
        df = pl.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        })
        result = validator.validate(df)
        assert result.valid is False
        assert any("kosong" in e.lower() for e in result.errors)

    def test_none_df_fails(self):
        validator = DataValidator()
        result = validator.validate(None)
        assert result.valid is False


class TestCreateValidatorFromConfig:
    def test_defaults(self):
        validator = create_validator_from_config({})
        assert validator.max_price_change_pct == 0.5
        assert validator.allow_gaps is False
        assert validator.min_volume == 0.0

    def test_custom_config(self):
        config = {
            "validation": {
                "max_price_change_pct": 0.8,
                "allow_gaps": True,
                "min_volume": 1.0,
            }
        }
        validator = create_validator_from_config(config)
        assert validator.max_price_change_pct == 0.8
        assert validator.allow_gaps is True
        assert validator.min_volume == 1.0
