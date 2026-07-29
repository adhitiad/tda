"""
Fractional Kelly Criterion position sizing.
"""

from __future__ import annotations

from typing import Any

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("kelly_sizing")


def calculate_kelly_fraction(
    probability: float,
    risk_reward_ratio: float,
    fraction: float = 0.5,
    max_risk_per_trade: float = 0.05,
) -> dict[str, Any]:
    """
    Calculate Kelly Criterion position sizing.

    Kelly formula: f* = (bp - q) / b
    - p = probability of success
    - q = probability of failure = 1 - p
    - b = win/loss ratio (risk/reward)

    :param probability: Probability of success between 0 and 1.
    :param risk_reward_ratio: Risk/Reward ratio (b).
    :param fraction: Fraction of Kelly to use (default Half-Kelly = 0.5).
    :param max_risk_per_trade: Maximum allowed risk per trade as fraction of capital.
    :return: Dict with kelly_fraction, recommended_margin_percentage, suggested_leverage.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError("probability must be between 0 and 1")
    if risk_reward_ratio <= 0:
        raise ValueError("risk_reward_ratio must be positive")

    p = probability
    q = 1.0 - p
    b = risk_reward_ratio

    kelly_full = (b * p - q) / b
    kelly_fraction = kelly_full * fraction

    if kelly_fraction < 0:
        kelly_fraction = 0.0

    kelly_fraction = min(kelly_fraction, max_risk_per_trade)
    recommended_margin_percentage = kelly_fraction * 100

    suggested_leverage = 1.0
    if recommended_margin_percentage > 0:
        if recommended_margin_percentage < 5:
            suggested_leverage = 5.0
        elif recommended_margin_percentage < 10:
            suggested_leverage = 3.0
        elif recommended_margin_percentage < 20:
            suggested_leverage = 2.0
        else:
            suggested_leverage = 1.0

    return {
        "kelly_fraction": round(float(kelly_fraction), 6),
        "recommended_margin_percentage": f"{recommended_margin_percentage:.2f}%",
        "suggested_leverage": f"{int(suggested_leverage)}x",
    }