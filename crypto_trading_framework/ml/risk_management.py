import polars as pl

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("risk_management")


def compute_atr_sl_tp(
    entry_price: float,
    atr: float,
    direction: str,
    sl_multiplier: float = 1.5,
    tp_multiplier: float = 2.0,
    trailing_stop_atr_multiple: float = 1.0,
) -> dict:
    if direction == "LONG":
        stop_loss = entry_price - (atr * sl_multiplier)
        take_profit = entry_price + (atr * tp_multiplier)
        trailing_step = atr * trailing_stop_atr_multiple
    else:
        stop_loss = entry_price + (atr * sl_multiplier)
        take_profit = entry_price - (atr * tp_multiplier)
        trailing_step = atr * trailing_stop_atr_multiple

    return {
        "stop_loss_atr": round(stop_loss, 2),
        "take_profit_atr": round(take_profit, 2),
        "trailing_step_atr": round(trailing_step, 2),
    }


def compute_kelly_criterion(
    probability: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    capital: float = 10000.0,
    max_risk_per_trade: float = 0.05,
    kelly_fraction: float = 0.5,
) -> dict:
    p = max(0.0, min(1.0, probability))
    q = 1.0 - p

    risk_per_unit = abs(entry_price - stop_loss)
    reward_per_unit = abs(take_profit - entry_price)

    if risk_per_unit <= 0 or reward_per_unit <= 0 or capital <= 0:
        return {
            "kelly_fraction": 0.0,
            "recommended_margin_percentage": "0%",
            "suggested_leverage": "1x",
            "position_size": 0.0,
        }

    b = reward_per_unit / risk_per_unit

    if b <= 0:
        kelly = 0.0
    else:
        kelly = (b * p - q) / b
        kelly = max(0.0, kelly)

    fractional_kelly = kelly * kelly_fraction
    max_risk_amount = capital * max_risk_per_trade
    recommended_margin = min(fractional_kelly * capital, max_risk_amount)
    recommended_margin_pct = recommended_margin / capital if capital > 0 else 0.0

    atr_pct = (risk_per_unit * 2.0) / entry_price if entry_price > 0 else 0.01
    raw_leverage = 0.015 / (atr_pct + 1e-9)
    leverage_val = max(1.0, min(10.0, round(raw_leverage)))
    leverage = f"{leverage_val:.0f}x"

    return {
        "kelly_fraction": round(fractional_kelly, 4),
        "recommended_margin_percentage": f"{recommended_margin_pct * 100:.1f}%",
        "suggested_leverage": leverage,
        "position_size": round(recommended_margin, 2),
    }


def enrich_signal_with_risk(
    signal: dict,
    result_df: pl.DataFrame,
    config: dict,
    capital: float = 10000.0,
) -> dict:
    rm_cfg = config.get("risk_management", {})
    sl_multiplier = rm_cfg.get("atr_multiplier_sl", 1.5)
    tp_multiplier = rm_cfg.get("atr_multiplier_tp", 2.0)
    trailing_stop_atr_multiple = rm_cfg.get("trailing_stop_atr_multiple", 1.0)
    kelly_enabled = rm_cfg.get("kelly_enabled", True)
    kelly_fraction = rm_cfg.get("kelly_fraction", 0.5)
    max_risk_per_trade = rm_cfg.get("max_risk_per_trade", 0.05)

    last_row = result_df.tail(1)
    atr = last_row["atr"].item() if "atr" in last_row.columns else 0.0
    if atr <= 0:
        atr = last_row["close"].item() * 0.02 if "close" in last_row.columns else 0.0

    entry_price = signal.get("entry", last_row["close"].item() if "close" in last_row.columns else 0.0)
    direction = signal.get("direction", "LONG")

    risk_levels = compute_atr_sl_tp(
        entry_price=entry_price,
        atr=atr,
        direction=direction,
        sl_multiplier=sl_multiplier,
        tp_multiplier=tp_multiplier,
        trailing_stop_atr_multiple=trailing_stop_atr_multiple,
    )

    signal["stop_loss"] = risk_levels["stop_loss_atr"]
    signal["take_profit"] = risk_levels["take_profit_atr"]
    signal["stop_loss_atr"] = risk_levels["stop_loss_atr"]
    signal["take_profit_atr"] = risk_levels["take_profit_atr"]
    signal["trailing_step_atr"] = risk_levels["trailing_step_atr"]
    signal["atr"] = round(atr, 2)

    position_sizing = {"kelly_fraction": 0.0, "recommended_margin_percentage": "0%", "suggested_leverage": "1x", "position_size": 0.0}
    if kelly_enabled:
        prob = signal.get("probability_float")
        if prob is None:
            raw_prob = signal.get("probability", 0.0)
            if isinstance(raw_prob, str):
                if "%" in raw_prob:
                    try:
                        prob = float(raw_prob.replace("%", "")) / 100.0
                    except Exception:
                        prob = 0.65
                else:
                    prob = 0.65
            else:
                prob = float(raw_prob)

        if prob > 0:
            position_sizing = compute_kelly_criterion(
                probability=prob,
                entry_price=entry_price,
                stop_loss=risk_levels["stop_loss_atr"],
                take_profit=risk_levels["take_profit_atr"],
                capital=capital,
                max_risk_per_trade=max_risk_per_trade,
                kelly_fraction=kelly_fraction,
            )

    signal["position_sizing"] = position_sizing
    return signal
