from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("strategies")


class BaseStrategy(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_signal(self, df: pl.DataFrame, config: dict[str, Any]) -> dict[str, Any] | None: ...

    @abstractmethod
    def validate(self, df: pl.DataFrame) -> bool: ...


class DefaultStrategy(BaseStrategy):
    def name(self) -> str:
        return "default"

    def generate_signal(self, df: pl.DataFrame, config: dict[str, Any]) -> dict[str, Any] | None:
        from crypto_trading_framework.ml.signals import generate_signal

        threshold = config.get("signal", {}).get("threshold", 0.55)
        return generate_signal(
            df,
            threshold=threshold,
            current_price=df["close"].tail(1).item(),
            min_adx=config.get("signal", {}).get("min_adx", 25.0),
            require_volume_spike=config.get("signal", {}).get("require_volume_spike", False),
        )

    def validate(self, df: pl.DataFrame) -> bool:
        return not df.is_empty() and len(df) > 60
