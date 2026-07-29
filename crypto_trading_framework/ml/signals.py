import polars as pl
from tabulate import tabulate

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("signals")


def generate_signal(
    df: pl.DataFrame,
    prob_col: str = "prob",
    threshold: float = 0.75,
    current_price: float | None = None,
    min_adx: float = 0.5,
    require_volume_spike: bool = False,
    atr_multiplier_sl: float = 1.5,
    atr_multiplier_tp: float = 2.0,
) -> dict | None:
    last_row = df.tail(1)
    prob = last_row[prob_col].item()
    close = last_row["close"].item()

    if current_price is None:
        current_price = close

    if prob < threshold:
        return None

    if prob >= 0.5:
        direction = "LONG"
    else:
        direction = "SHORT"

    if "regime_trending" in last_row.columns and last_row["regime_trending"].item() == 0:
        logger.info(f"[FILTER] Sinyal {direction} difilter: bukan regime trending")
        return None

    if "adx" in last_row.columns and last_row["adx"].item() < min_adx:
        logger.info(f"[FILTER] Sinyal {direction} difilter: ADX={last_row['adx'].item():.3f} < {min_adx}")
        return None

    if require_volume_spike:
        if "volume_spike_flag" in last_row.columns and last_row["volume_spike_flag"].item() == 0:
            logger.info(f"[FILTER] Sinyal {direction} difilter: tidak ada volume spike")
            return None

    atr = last_row["atr"].item() if "atr" in last_row.columns else close * 0.02

    if direction == "LONG":
        tp = round(current_price + atr * atr_multiplier_tp, 2)
        sl = round(current_price - atr * atr_multiplier_sl, 2)
    else:
        tp = round(current_price - atr * atr_multiplier_tp, 2)
        sl = round(current_price + atr * atr_multiplier_sl, 2)

    reasons = []
    if "rsi" in last_row.columns:
        rsi_val = last_row["rsi"].item()
        reasons.append(f"RSI={rsi_val:.1f}")
    if "stoch_k" in last_row.columns:
        stoch_val = last_row["stoch_k"].item()
        reasons.append(f"StochK={stoch_val:.1f}")
    if "ema_20" in last_row.columns and "ema_50" in last_row.columns:
        ema20 = last_row["ema_20"].item()
        ema50 = last_row["ema_50"].item()
        if direction == "LONG" and ema20 > ema50:
            reasons.append("EMA20>EMA50")
        elif direction == "SHORT" and ema20 < ema50:
            reasons.append("EMA20<EMA50")
    if "adx" in last_row.columns:
        adx_val = last_row["adx"].item()
        reasons.append(f"ADX={adx_val:.2f}")
    if "volume_spike_flag" in last_row.columns and last_row["volume_spike_flag"].item() == 1:
        reasons.append("VolumeSpike")

    reason_str = "; ".join(reasons) if reasons else "Model prediction"

    signal_type = "Spot" if direction == "LONG" else "Futures"
    action = "BUY" if direction == "LONG" else "SELL"

    signal = {
        "direction": direction,
        "signal_type": signal_type,
        "action": action,
        "probability": f"{prob*100:.1f}%",
        "probability_float": round(prob, 4),
        "entry": current_price,
        "take_profit": tp,
        "stop_loss": sl,
        "reason": reason_str,
        "winrate": "Estimasi: ~65-75% (butuh backtest)",
        "atr": round(atr, 2),
    }
    return signal


def generate_multi_tf_signal(
    primary_df: pl.DataFrame,
    auxiliary_dfs: dict[str, pl.DataFrame],
    threshold: float = 0.75,
    min_confirmations: int = 2,
) -> dict | None:
    primary_signal = generate_signal(primary_df, threshold=threshold)
    if primary_signal is None:
        return None

    confirmations = 0
    if primary_signal["direction"] == "LONG":
        for tf, df in auxiliary_dfs.items():
            if "ema_20" in df.columns and "ema_50" in df.columns:
                ema20 = df.tail(1)["ema_20"].item()
                ema50 = df.tail(1)["ema_50"].item()
                if ema20 > ema50:
                    confirmations += 1
    else:
        for tf, df in auxiliary_dfs.items():
            if "ema_20" in df.columns and "ema_50" in df.columns:
                ema20 = df.tail(1)["ema_20"].item()
                ema50 = df.tail(1)["ema_50"].item()
                if ema20 < ema50:
                    confirmations += 1

    if confirmations >= min_confirmations:
        primary_signal["reason"] += f"; MTF confirm={confirmations}"
        return primary_signal

    logger.info(f"[FILTER] Sinyal {primary_signal['direction']} difilter: konfirmasi MTF={confirmations}/{min_confirmations}")
    return None


def print_signal_table(signals: list):
    if not signals:
        logger.info("")
        logger.info("[TIDAK ADA SINYAL] Tidak ada sinyal trading yang memenuhi threshold.")
        return

    headers = ["Pair", "Timeframe", "Direction", "Action", "Type", "Prob", "Entry", "TP", "SL", "Reason"]
    rows = []
    for sig in signals:
        rows.append([
            sig.get("symbol", "N/A"),
            sig.get("timeframe", "N/A"),
            sig["direction"],
            sig.get("action", "N/A"),
            sig.get("signal_type", "N/A"),
            sig["probability"],
            sig["entry"],
            sig["take_profit"],
            sig["stop_loss"],
            sig["reason"][:50],
        ])

    logger.info("")
    logger.info("=" * 120)
    logger.info("SINYAL TRADING")
    logger.info("=" * 120)
    logger.info("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
    logger.info("=" * 120)
