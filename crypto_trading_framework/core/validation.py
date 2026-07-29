import polars as pl

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("validation")


class ValidationResult:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.valid = True

    def add_error(self, message: str):
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str):
        self.warnings.append(message)

    def __repr__(self):
        status = "PASS" if self.valid else "FAIL"
        parts = [f"ValidationResult(status={status}"]
        if self.errors:
            parts.append(f"errors={self.errors}")
        if self.warnings:
            parts.append(f"warnings={self.warnings}")
        return ", ".join(parts) + ")"


class DataValidator:
    REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(
        self,
        max_price_change_pct: float = 0.5,
        allow_gaps: bool = False,
        min_volume: float = 0.0,
    ):
        self.max_price_change_pct = max_price_change_pct
        self.allow_gaps = allow_gaps
        self.min_volume = min_volume

    def validate(self, df: pl.DataFrame) -> ValidationResult:
        result = ValidationResult()

        if df is None or df.is_empty():
            result.add_error("DataFrame kosong atau None")
            return result

        self._check_schema(df, result)
        self._check_types(df, result)
        self._check_nulls(df, result)
        self._check_ranges(df, result)
        self._check_outliers(df, result)
        self._check_volume(df, result)
        self._check_gaps(df, result)

        return result

    def _check_schema(self, df: pl.DataFrame, result: ValidationResult):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            result.add_error(f"Kolom yang hilang: {missing}")

    def _check_types(self, df: pl.DataFrame, result: ValidationResult):
        if "timestamp" in df.columns:
            ts_dtype = df.schema["timestamp"]
            if not (isinstance(ts_dtype, pl.Datetime) or isinstance(ts_dtype, pl.Int64)):
                result.add_error(
                    f"Kolom timestamp harus Datetime atau Int64, dapat {ts_dtype}"
                )

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                dtype = df.schema[col]
                if not (
                    isinstance(dtype, pl.Float64)
                    or isinstance(dtype, pl.Float32)
                    or isinstance(dtype, pl.Int64)
                    or isinstance(dtype, pl.Int32)
                ):
                    result.add_error(f"Kolom {col} harus numerik, dapat {dtype}")

    def _check_nulls(self, df: pl.DataFrame, result: ValidationResult):
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    result.add_error(f"Kolom {col} memiliki {null_count} nilai null")

    def _check_ranges(self, df: pl.DataFrame, result: ValidationResult):
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            invalid_high = (df["high"] < df["low"]).sum()
            if invalid_high > 0:
                result.add_error(
                    f"{invalid_high} baris memiliki high < low"
                )

            invalid_open = (df["open"] > df["high"]).sum()
            if invalid_open > 0:
                result.add_error(
                    f"{invalid_open} baris memiliki open > high"
                )

            invalid_close = (df["close"] > df["high"]).sum()
            if invalid_close > 0:
                result.add_error(
                    f"{invalid_close} baris memiliki close > high"
                )

            invalid_open_low = (df["open"] < df["low"]).sum()
            if invalid_open_low > 0:
                result.add_error(
                    f"{invalid_open_low} baris memiliki open < low"
                )

            invalid_close_low = (df["close"] < df["low"]).sum()
            if invalid_close_low > 0:
                result.add_error(
                    f"{invalid_close_low} baris memiliki close < low"
                )

    def _check_outliers(self, df: pl.DataFrame, result: ValidationResult):
        if "close" not in df.columns or "timestamp" not in df.columns:
            return

        df_sorted = df.sort("timestamp")
        if df_sorted.height < 2:
            return

        close = df_sorted["close"]
        pct_change = (close / close.shift(1)) - 1.0
        pct_change = pct_change.abs()

        outlier_mask = pct_change > self.max_price_change_pct
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            result.add_warning(
                f"{outlier_count} outlier harga terdeteksi (perubahan > {self.max_price_change_pct:.0%})"
            )

    def _check_volume(self, df: pl.DataFrame, result: ValidationResult):
        if "volume" not in df.columns:
            return

        negative_volume = (df["volume"] < self.min_volume).sum()
        if negative_volume > 0:
            result.add_error(
                f"{negative_volume} baris memiliki volume < {self.min_volume}"
            )

    def _check_gaps(self, df: pl.DataFrame, result: ValidationResult):
        if self.allow_gaps:
            return

        if "timestamp" not in df.columns:
            return

        df_sorted = df.sort("timestamp")
        if df_sorted.height < 2:
            return

        ts = df_sorted["timestamp"]

        if isinstance(ts.dtype, pl.Datetime):
            diffs = ts.diff().dt.total_minutes()
        elif isinstance(ts.dtype, pl.Int64):
            diffs = ts.diff()
        else:
            return

        gap_mask = diffs.is_null().not_() & (diffs > diffs.median() * 3)
        gap_count = gap_mask.sum()

        if gap_count > 0:
            result.add_warning(
                f"{gap_count} gap timestamp terdeteksi"
            )


def create_validator_from_config(config: dict | None = None) -> DataValidator:
    validation_cfg = config.get("validation", {}) if config else {}
    return DataValidator(
        max_price_change_pct=float(validation_cfg.get("max_price_change_pct", 0.5)),
        allow_gaps=bool(validation_cfg.get("allow_gaps", False)),
        min_volume=float(validation_cfg.get("min_volume", 0.0)),
    )
