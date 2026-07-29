import polars as pl
from typing import Optional

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("rule_signals")


def generate_rule_based_signal(
    df: pl.DataFrame,
    min_confluences: int = 3,
) -> Optional[dict]:
    last = df.tail(1)
    if len(last) == 0:
        return None

    direction = None
    score = 0
    reasons = []

    ema_bullish = False
    ema_bearish = False
    if "ema_20" in last.columns and "ema_50" in last.columns:
        ema20 = last["ema_20"].item()
        ema50 = last["ema_50"].item()
        if ema20 > ema50:
            ema_bullish = True
        else:
            ema_bearish = True

    if ema_bullish:
        score += 1
        reasons.append("EMA20>EMA50")
    elif ema_bearish:
        score += 1
        reasons.append("EMA20<EMA50")

    rsi_bullish = False
    rsi_bearish = False
    if "rsi" in last.columns:
        rsi = last["rsi"].item()
        if rsi > 50 and rsi < 70:
            rsi_bullish = True
        elif rsi < 50 and rsi > 30:
            rsi_bearish = True

    if rsi_bullish:
        score += 1
        reasons.append(f"RSI={rsi:.1f}")
    elif rsi_bearish:
        score += 1
        reasons.append(f"RSI={rsi:.1f}")

    macd_bullish = False
    macd_bearish = False
    if "macd_hist" in last.columns:
        macd_hist = last["macd_hist"].item()
        if macd_hist > 0:
            macd_bullish = True
        else:
            macd_bearish = True

    if macd_bullish:
        score += 1
        reasons.append("MACD>0")
    elif macd_bearish:
        score += 1
        reasons.append("MACD<0")

    stoch_bullish = False
    stoch_bearish = False
    if "stoch_k" in last.columns and "stoch_d" in last.columns:
        stoch_k = last["stoch_k"].item()
        stoch_d = last["stoch_d"].item()
        if stoch_k > stoch_d and stoch_k < 80:
            stoch_bullish = True
        elif stoch_k < stoch_d and stoch_k > 20:
            stoch_bearish = True

    if stoch_bullish:
        score += 1
        reasons.append("StochK>StochD")
    elif stoch_bearish:
        score += 1
        reasons.append("StochK<StochD")

    vwap_bullish = False
    vwap_bearish = False
    if "vwap_deviation" in last.columns:
        vwap_dev = last["vwap_deviation"].item()
        if vwap_dev > 0:
            vwap_bullish = True
        else:
            vwap_bearish = True

    if vwap_bullish:
        score += 1
        reasons.append("Price>VWAP")
    elif vwap_bearish:
        score += 1
        reasons.append("Price<VWAP")

    volume_bullish = False
    volume_bearish = False
    if "volume_spike_flag" in last.columns:
        spike = last["volume_spike_flag"].item()
        if spike == 1:
            if ema_bullish or macd_bullish:
                volume_bullish = True
            elif ema_bearish or macd_bearish:
                volume_bearish = True

    if volume_bullish:
        score += 1
        reasons.append("VolumeSpike+")
    elif volume_bearish:
        score += 1
        reasons.append("VolumeSpike-")

    if ema_bullish and rsi_bullish and macd_bullish:
        direction = "LONG"
    elif ema_bearish and rsi_bearish and macd_bearish:
        direction = "SHORT"
    elif score >= min_confluences:
        if ema_bullish or rsi_bullish or macd_bullish:
            direction = "LONG"
        else:
            direction = "SHORT"

    if direction and score >= min_confluences:
        close = last["close"].item()
        atr = last["atr"].item() if "atr" in last.columns else close * 0.02

        if direction == "LONG":
            tp = round(close + atr * 2, 2)
            sl = round(close - atr * 1, 2)
        else:
            tp = round(close - atr * 2, 2)
            sl = round(close + atr * 1, 2)

        action = "BUY" if direction == "LONG" else "SELL"
        signal = {
            "direction": direction,
            "signal_type": "RuleBased",
            "action": action,
            "probability": f"{score}/{min_confluences}+ confluence",
            "entry": close,
            "take_profit": tp,
            "stop_loss": sl,
            "reason": "; ".join(reasons),
            "winrate": "Estimasi: ~60-70% (rule-based)",
            "score": score,
        }
        return signal

    return None
