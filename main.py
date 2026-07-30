import asyncio
import gc
import os
import random
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import polars as pl
import psutil
import torch
import yaml
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from crypto_trading_framework.core.bot import AutomatedTradingBot
from crypto_trading_framework.core.config_schema import validate_config
from crypto_trading_framework.core.indicators import add_all_indicators
from crypto_trading_framework.core.kill_switch import is_active as kill_switch_is_active
from crypto_trading_framework.core.logging import get_logger
from crypto_trading_framework.data_ingestion import DataIngestion
from crypto_trading_framework.db.database import create_schema
from crypto_trading_framework.ml.continuous_learning import FeedbackLoopEngine
from crypto_trading_framework.ml.inference_worker import get_celery_app
from crypto_trading_framework.ml.ml_pipeline import MLPipeline
from crypto_trading_framework.ml.model import create_model
from crypto_trading_framework.ml.multi_timeframe import fuse_multi_timeframe
from crypto_trading_framework.ml.risk_management import enrich_signal_with_risk
from crypto_trading_framework.ml.rule_signals import generate_rule_based_signal
from crypto_trading_framework.ml.self_tuning import SelfTuningEngine
from crypto_trading_framework.ml.sentiment import SentimentEngine
from crypto_trading_framework.ml.shadow_trader import ShadowTrader
from crypto_trading_framework.ml.signals import generate_signal, print_signal_table
from crypto_trading_framework.ml.smart_money_tracker import SmartMoneyTracker

logger = get_logger("quantuis")


def _interpolate_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        pattern = re.compile(r'\$\{([^}]+)\}')
        def replacer(match: re.Match) -> str:
            env_var = match.group(1)
            default = ""
            if ":" in env_var:
                env_var, default = env_var.split(":", 1)
            return os.getenv(env_var, default)
        return pattern.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    raw_config = _interpolate_env_vars(raw_config)
    return validate_config(raw_config)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_memory_usage(stage: str):
    try:
        process = psutil.Process()
        ram_info = process.memory_info()
        total_ram = psutil.virtual_memory().total
        used_ram_gb = ram_info.rss / (1024 ** 3)
        total_ram_gb = total_ram / (1024 ** 3)
        ram_pct = (ram_info.rss / total_ram) * 100
        gpu_mem = ""
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
            gpu_mem = f" | GPU allocated: {gpu_allocated:.1f}MB, reserved: {gpu_reserved:.1f}MB"
        logger.info(f"[Memory Audit] {stage}: RAM {used_ram_gb:.2f}GB / {total_ram_gb:.2f}GB ({ram_pct:.1f}%){gpu_mem}")
    except Exception as e:
        logger.debug(f"[Memory Audit] Failed to get memory info: {e}")


def merge_market_data(df: pl.DataFrame | pl.LazyFrame, market_data: dict) -> pl.DataFrame | pl.LazyFrame:
    for md_df in market_data.values():
        if md_df is not None and not md_df.is_empty():
            for col in md_df.columns:
                if col != "timestamp" and col not in df.columns:
                    try:
                        val = md_df[col].tail(1).item()
                    except Exception:
                        try:
                            val = float(md_df[col].mean())
                        except Exception:
                            continue
                    df = df.with_columns(pl.lit(val).alias(col))
    return df


# Global state
current_signals: dict[str, dict[str, Any]] = {}
loaded_models: dict[str, Any] = {}
feedback_engine: FeedbackLoopEngine | None = None
monitoring_task: asyncio.Task | None = None
learning_task: asyncio.Task | None = None
shadow_trader: ShadowTrader | None = None
position_monitor_task: asyncio.Task | None = None
sentiment_engine: SentimentEngine | None = None
self_tuning_engine: SelfTuningEngine | None = None
smart_money_tracker: SmartMoneyTracker | None = None
celery_app: Any = None
app_start_time: float = 0.0


def _load_models_from_disk(models_dir: str = "models") -> None:
    models_path = Path(models_dir)
    if not models_path.exists():
        logger.warning(f"[Quantuis] Models directory {models_dir} not found. Skipping model loading.")
        return

    best_weights = models_path / "best_weights.pth"
    scaler_path = models_path / "scaler.pkl"
    golden_memory_path = models_path / "golden_memory.json"

    if best_weights.exists():
        try:
            loaded_models["best_weights"] = str(best_weights)
            logger.info(f"[Quantuis] Loaded model weights from {best_weights}")
        except Exception as e:
            logger.error(f"[Quantuis] Failed to load model weights: {e}")

    if scaler_path.exists():
        try:
            loaded_models["scaler"] = str(scaler_path)
            logger.info(f"[Quantuis] Loaded scaler from {scaler_path}")
        except Exception as e:
            logger.error(f"[Quantuis] Failed to load scaler: {e}")

    if golden_memory_path.exists():
        try:
            loaded_models["golden_memory"] = str(golden_memory_path)
            logger.info(f"[Quantuis] Loaded golden memory from {golden_memory_path}")
        except Exception as e:
            logger.error(f"[Quantuis] Failed to load golden memory: {e}")

    logger.info(f"[Quantuis] Total models loaded: {len(loaded_models)}")


async def _fetch_coin_data(symbol: str, config: dict) -> dict[str, pl.DataFrame] | None:
    ingestion = DataIngestion(
        exchange_id=config.get("data", {}).get("exchange_id", "okx"),
        symbol=symbol,
    )
    try:
        timeframes = config.get("data", {}).get("timeframes", ["m15", "h1", "h4"])
        lookback = config.get("data", {}).get("lookback", 2000)
        raw_data = await ingestion.fetch_multi_timeframe(timeframes=timeframes, limit=lookback)

        if not raw_data:
            logger.warning(f"[Quantuis] No CCXT data for {symbol}, trying yfinance fallback")
            yf_tickers = config.get("data", {}).get("yfinance_ticker", [])
            yf_ticker = next((t for t in yf_tickers if t.replace("-", "").replace("=", "") in symbol), None)
            if yf_ticker and config.get("data", {}).get("fallback_enabled", True):
                raw_data = {}
                interval_map = {"m5": "5m", "m15": "15m", "m30": "30m", "h1": "1h", "h4": "4h", "d1": "1d"}
                for tf in timeframes:
                    df_yf = ingestion.fetch_yfinance(
                        ticker=yf_ticker,
                        period=config.get("data", {}).get("yfinance_period", "10y"),
                        interval=interval_map.get(tf, "1h"),
                    )
                    if df_yf is not None:
                        raw_data[tf] = df_yf

        await ingestion.close()
        return raw_data if raw_data else None
    except Exception:
        logger.exception(f"[Quantuis] Error fetching data for {symbol}")
        try:
            await ingestion.close()
        except Exception:
            logger.exception("[Quantuis] Error closing ingestion")
        return None


async def _fetch_macro_data(config: dict) -> dict[str, pl.DataFrame]:
    market_data: dict[str, pl.DataFrame] = {}
    try:
        ingestion = DataIngestion(
            exchange_id=config.get("data", {}).get("exchange_id", "okx"),
            symbol="BTC/USDT:USDT",
        )

        md_config = config.get("market_data", {})
        if md_config.get("orderbook", {}).get("enabled", False):
            ob = await ingestion.fetch_orderbook_async(limit=md_config["orderbook"].get("limit", 100))
            if ob is not None:
                market_data["orderbook"] = ob
                logger.info(f"[Quantuis] Orderbook: imbalance={ob['bid_ask_imbalance'].item():.4f}")

        if md_config.get("funding_rate", {}).get("enabled", False):
            fr = await ingestion.fetch_funding_rate_async()
            if fr is not None:
                market_data["funding_rate"] = fr
                logger.info(f"[Quantuis] Funding rate: {fr['funding_rate'].item():.6f}")

        if md_config.get("open_interest", {}).get("enabled", False):
            oi = await ingestion.fetch_open_interest_async()
            if oi is not None:
                market_data["open_interest"] = oi
                logger.info(f"[Quantuis] Open interest: {oi['open_interest'].item():.2f}")

        await ingestion.close()
    except Exception as e:
        logger.error(f"[Quantuis] Error fetching market data: {e}")

    yf_tickers = config.get("data", {}).get("yfinance_ticker", [])
    for ticker in yf_tickers[:2]:
        try:
            ingestion = DataIngestion(
                exchange_id=config.get("data", {}).get("exchange_id", "okx"),
                symbol="BTC/USDT:USDT",
            )
            df_yf = ingestion.fetch_yfinance(
                ticker=ticker,
                period="1y",
                interval="1d",
            )
            if df_yf is not None and not df_yf.is_empty():
                market_data[f"macro_{ticker}"] = df_yf
                logger.info(f"[Quantuis] Loaded macro data for {ticker}")
            await ingestion.close()
        except Exception as e:
            logger.error(f"[Quantuis] Error fetching macro data for {ticker}: {e}")

    return market_data


async def _process_coin(symbol: str, config: dict) -> None:
    try:
        logger.info(f"[Quantuis] Processing {symbol}...")
        _log_memory_usage(f"Start {symbol}")

        raw_data = await _fetch_coin_data(symbol, config)
        if not raw_data:
            logger.warning(f"[Quantuis] No data for {symbol}, skipping")
            return

        market_data = await _fetch_macro_data(config)

        fusion_enabled = config.get("multi_timeframe", {}).get("enabled", False)
        primary_tf = config.get("multi_timeframe", {}).get("primary_timeframe", "m15")

        if fusion_enabled:
            processed_data = {}
            for tf, df_raw in raw_data.items():
                df = add_all_indicators(df_raw.lazy())
                processed_data[tf] = df
            fused_df = fuse_multi_timeframe(processed_data, primary_timeframe=primary_tf)
            raw_data_fused = {primary_tf: fused_df}
            del processed_data, fused_df
        else:
            raw_data_fused = raw_data

        for tf, df_raw in raw_data_fused.items():
            df = add_all_indicators(df_raw.lazy()) if not fusion_enabled else df_raw
            df = merge_market_data(df, market_data)
            df = df.collect()
            pipeline = MLPipeline(scaler_type=config["ml"]["scaler_type"])
            target_type = config.get("target", {}).get("type", "binary")
            df = pipeline.define_target(df, forward_periods=config["ml"]["forward_periods"], target_type=target_type)
            df = df.drop_nulls()

            if df.is_empty():
                logger.warning(f"[Quantuis] Data {tf} kosong untuk {symbol}")
                continue

            features, _ = pipeline.prepare_features(df, feature_cols=config["ml"]["feature_cols"])
            targets = df.select("target").to_numpy().flatten()

            time_steps = config["ml"]["time_steps"]
            if len(features) <= time_steps:
                logger.warning(f"[Quantuis] Data {tf} terlalu sedikit untuk {symbol}")
                continue

            split_idx = int(len(features) * (1 - config["ml"]["test_size"]))
            features_train_raw = features[:split_idx]
            features_test_raw = features[max(0, split_idx - time_steps):]
            targets_train_raw = targets[:split_idx]
            targets_test_raw = targets[max(0, split_idx - time_steps):]

            scaled_train = pipeline.scale_features(features_train_raw, fit=True)
            scaled_test = pipeline.scale_features(features_test_raw, fit=False)

            if len(scaled_train) <= time_steps or len(scaled_test) <= time_steps:
                logger.warning(f"[Quantuis] Data {tf} terlalu sedikit setelah split untuk {symbol}")
                continue

            X_train, _ = pipeline.create_sequences(scaled_train, targets_train_raw, time_steps=time_steps)
            X_test, _ = pipeline.create_sequences(scaled_test, targets_test_raw, time_steps=time_steps)

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            input_size = X_train.shape[2]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_type = config["model"]["type"]
            if model_type == "ensemble":
                model = create_model(
                    model_type="lstm",
                    input_size=input_size,
                ).to(device)
            else:
                model = create_model(
                    model_type=model_type,
                    input_size=input_size,
                ).to(device)

            best_weights_path = Path("models") / "best_weights.pth"
            state_dict = None
            if best_weights_path.exists():
                try:
                    state_dict = torch.load(str(best_weights_path), map_location=device, weights_only=True)
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"[Quantuis] Loaded weights for {symbol}")
                except Exception:
                    pass

            # Run inference via Celery
            task = celery_app.send_task(
                "inference.run",
                args=[
                    str(best_weights_path),
                    model_type if model_type != "ensemble" else "lstm",
                    input_size,
                    X_test.tolist(),
                    str(device),
                ],
                queue="inference",
            )
            # Wait for result with timeout
            timeout = config.get("task_queue", {}).get("timeout", 120)
            try:
                preds_list = await asyncio.get_event_loop().run_in_executor(
                    None, lambda t=task, timeout=timeout: t.get(timeout=timeout)
                )
                preds = np.array(preds_list, dtype=np.float32)
            except Exception as e:
                logger.error(f"[Quantuis] Inference task failed: {e}")
                del model, state_dict, best_weights_path
                del X_train, X_test, features_train_raw, features_test_raw
                del scaled_train, scaled_test, targets_train_raw, targets_test_raw
                del features, targets, pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue

            result_df = df.tail(len(preds)).with_columns(pl.Series("prob", preds))

            threshold = config.get("signal", {}).get("threshold", 0.55)
            rm_cfg = config.get("risk_management", {})
            signal = generate_signal(
                result_df,
                threshold=threshold,
                current_price=df["close"].tail(1).item(),
                min_adx=config.get("signal", {}).get("min_adx", 25.0),
                require_volume_spike=config.get("signal", {}).get("require_volume_spike", False),
                atr_multiplier_sl=rm_cfg.get("atr_multiplier_sl", 1.5),
                atr_multiplier_tp=rm_cfg.get("atr_multiplier_tp", 2.0),
            )

            if signal is None and config.get("signal", {}).get("use_rule_fallback", False):
                signal = generate_rule_based_signal(df, min_confluences=config.get("signal", {}).get("min_confluences", 3))

            # Free large temporary variables after inference
            del model, state_dict, best_weights_path
            del X_train, X_test, features_train_raw, features_test_raw
            del scaled_train, scaled_test, targets_train_raw, targets_test_raw
            del features, targets, pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if signal:
                signal["symbol"] = symbol
                signal["timeframe"] = tf

                # Smart Money data fetch and VETO logic
                smart_money_data: dict[str, Any] = {
                    "composite_signal": "NEUTRAL",
                    "veto_status": "NONE",
                }
                vetoed = False
                veto_reason = ""

                # Check oversold status from RSI
                is_oversold = False
                try:
                    if "rsi" in result_df.columns:
                        rsi_val = result_df["rsi"].tail(1).item()
                        is_oversold = rsi_val < 30
                except Exception:
                    pass

                # Smart Money VETO (DefiLlama/Coinglass) - only if not already vetoed
                if not vetoed and smart_money_tracker is not None and signal.get("direction") in ("LONG", "SHORT"):
                    try:
                        base_symbol = symbol.split("/")[0].upper()
                        sm_snapshot = await smart_money_tracker.get_smart_money_snapshot(base_symbol)
                        sm_vetoed, sm_reason, sm_mode = smart_money_tracker.evaluate_veto(
                            signal["direction"], sm_snapshot, is_oversold
                        )
                        smart_money_data = {
                            "cex_signal": sm_snapshot.get("cex_transparency", {}).get("signal", "NEUTRAL"),
                            "oi_signal": sm_snapshot.get("open_interest", {}).get("signal", "NEUTRAL"),
                            "liq_signal": sm_snapshot.get("liquidations", {}).get("signal", "NEUTRAL"),
                            "composite_signal": sm_snapshot.get("composite_signal", "NEUTRAL"),
                            "veto_status": "NONE",
                        }
                        if sm_vetoed:
                            vetoed = True
                            veto_reason = sm_reason
                            smart_money_data["veto_status"] = f"VETO_{sm_mode}"
                            logger.warning(f"[Quantuis] Smart Money VETO triggered ({sm_mode}): {sm_reason}")
                            try:
                                telegram_logger = get_logger("telegram")
                                telegram_logger.error(f"SMART_MONEY_VETO: {sm_reason}")
                            except Exception:
                                pass
                    except Exception:
                        logger.exception("[Quantuis] Error during smart money veto evaluation")

                if vetoed:
                    current_price = df["close"].tail(1).item()
                    current_signals[symbol] = {
                        "direction": signal.get("direction", "LONG"),
                        "action": "HOLD",
                        "status": "vetoed",
                        "type_simbol": "Crypto",
                        "simbol": symbol,
                        "signal": "VETOED",
                        "probability": signal.get("probability", "0%"),
                        "probability_float": signal.get("probability_float", 0.0),
                        "entry_zone": [signal.get("entry", 0.0)],
                        "current_price": current_price,
                        "take_profit": [signal.get("take_profit", 0.0)],
                        "stop_loss_atr": signal.get("stop_loss", 0.0),
                        "trailing_step_atr": signal.get("trailing_step_atr", 0.0),
                        "position_sizing": {"kelly_fraction": 0.0, "recommended_margin_percentage": "0%", "suggested_leverage": "1x", "position_size": 0.0},
                        "atr": signal.get("atr", 0.0),
                        "timeframe": tf,
                        "reason": veto_reason or "Vetoed by on-chain analysis",
                        "winrate": "",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "smart_money_analysis": smart_money_data,
                    }
                    if telegram_bot is not None:
                        try:
                            cs = current_signals[symbol]
                            entry_zone = cs.get("entry_zone", [0.0])
                            take_profit = cs.get("take_profit", [0.0])
                            stop_loss = cs.get("stop_loss_atr", 0.0)
                            await telegram_bot.send_signal(
                                symbol=symbol,
                                asset_type="crypto",
                                signal_data={
                                    "symbol": symbol,
                                    "simbol": symbol,
                                    "direction": cs.get("direction", "LONG"),
                                    "action": "HOLD",
                                    "entry": entry_zone[0] if isinstance(entry_zone, list) else entry_zone,
                                    "entry_zone": entry_zone,
                                    "current_price": current_price,
                                    "take_profit": take_profit if isinstance(take_profit, list) else [take_profit],
                                    "stop_loss_atr": stop_loss,
                                    "stop_loss": stop_loss,
                                    "timeframe": cs.get("timeframe", tf),
                                    "probability": cs.get("probability", "0%"),
                                    "probability_float": cs.get("probability_float", 0.0),
                                    "confidence": "LOW",
                                    "winrate": "",
                                    "win_rate": 0.0,
                                    "atr": cs.get("atr", 0.0),
                                    "reason": cs.get("reason", veto_reason or "Vetoed by on-chain analysis"),
                                    "indicators": {},
                                },
                            )
                        except Exception as e:
                            logger.error(f"[Quantuis] Error sending Telegram veto signal for {symbol}: {e}")
                else:
                    signal = enrich_signal_with_risk(
                        signal=signal,
                        result_df=result_df,
                        config=config,
                        capital=float(config.get("backtest", {}).get("initial_capital", 10000.0)),
                    )

                    entry_price = signal.get("entry", 0.0)
                    sl_price = signal.get("stop_loss", 0.0)
                    tp_price = signal.get("take_profit", 0.0)
                    sl_pct = abs(entry_price - sl_price) / entry_price if entry_price > 0 else 0.0
                    tp_pct = abs(tp_price - entry_price) / entry_price if entry_price > 0 else 0.0
                    current_price = df["close"].tail(1).item()

                    current_signals[symbol] = {
                        "direction": signal.get("direction", "LONG"),
                        "action": signal.get("action", "BUY"),
                        "status": "success",
                        "type_simbol": "Crypto",
                        "simbol": symbol,
                        "signal": "BULLISH" if signal.get("direction", "LONG") == "LONG" else "BEARISH",
                        "probability": signal.get("probability", "0%"),
                        "probability_float": signal.get("probability_float", 0.0),
                        "entry_zone": [signal.get("entry", 0.0)],
                        "current_price": current_price,
                        "take_profit": [tp_price],
                        "stop_loss_atr": sl_price,
                        "trailing_step_atr": signal.get("trailing_step_atr", 0.0),
                        "position_sizing": signal.get("position_sizing", {}),
                        "atr": signal.get("atr", 0.0),
                        "timeframe": tf,
                        "reason": signal.get("reason", ""),
                        "winrate": signal.get("winrate", ""),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "smart_money_analysis": smart_money_data,
                    }
                    logger.info(f"[Quantuis] Signal generated for {symbol}: {signal['direction']} @ {signal.get('entry', 0)}")

                    if telegram_bot is not None:
                        try:
                            cs = current_signals[symbol]
                            entry_zone = cs.get("entry_zone", [0.0])
                            take_profit = cs.get("take_profit", [0.0])
                            stop_loss = cs.get("stop_loss_atr", 0.0)
                            winrate_str = cs.get("winrate", "")
                            win_rate_val = 0.0
                            if winrate_str and "%" in winrate_str:
                                import re as _re
                                match = _re.search(r"(\d+)\s*-\s*(\d+)%", winrate_str)
                                if match:
                                    low = float(match.group(1))
                                    high = float(match.group(2))
                                    win_rate_val = (low + high) / 2.0
                            await telegram_bot.send_signal(
                                symbol=symbol,
                                asset_type="crypto",
                                signal_data={
                                    "symbol": symbol,
                                    "simbol": symbol,
                                    "direction": cs.get("direction", "LONG"),
                                    "action": cs.get("action", "BUY"),
                                    "entry": entry_zone[0] if isinstance(entry_zone, list) else entry_zone,
                                    "entry_zone": entry_zone,
                                    "current_price": current_price,
                                    "take_profit": take_profit if isinstance(take_profit, list) else [take_profit],
                                    "stop_loss_atr": stop_loss,
                                    "stop_loss": stop_loss,
                                    "timeframe": cs.get("timeframe", tf),
                                    "probability": cs.get("probability", "0%"),
                                    "probability_float": cs.get("probability_float", 0.0),
                                    "confidence": cs.get("probability", "MEDIUM"),
                                    "winrate": winrate_str,
                                    "win_rate": win_rate_val,
                                    "atr": cs.get("atr", 0.0),
                                    "reason": cs.get("reason", ""),
                                    "indicators": {},
                                },
                            )
                        except Exception as e:
                            logger.error(f"[Quantuis] Error sending Telegram signal for {symbol}: {e}")

                    if shadow_trader is not None:
                        try:
                            shadow_trader.execute_signal(current_signals[symbol])
                        except Exception as e:
                            logger.error(f"[Quantuis] Error executing shadow trade for {symbol}: {e}")

                    indicator_values: dict[str, float] = {}
                    for col in result_df.columns:
                        if col in ("prob", "target", "timestamp"):
                            continue
                        try:
                            val = result_df[col].tail(1).item()
                            if isinstance(val, (int, float)) and not isinstance(val, bool):
                                indicator_values[col] = float(val)
                        except Exception:
                            pass

                    if feedback_engine is not None:
                        try:
                            feedback_engine.store_signal(
                                indicator_values=indicator_values,
                                direction=signal.get("direction", "LONG"),
                                entry_price=signal.get("entry", 0.0),
                                probability=signal.get("probability_float", 0.0),
                                tp_pct=tp_pct,
                                sl_pct=sl_pct,
                                timeframe=tf,
                                symbol=symbol,
                            )
                        except Exception as e:
                            logger.error(f"[Quantuis] Error storing signal for feedback: {e}")
            else:
                current_signals[symbol] = {
                    "direction": "NEUTRAL",
                    "action": "HOLD",
                    "status": "pending",
                    "type_simbol": "Crypto",
                    "simbol": symbol,
                    "signal": "HOLD",
                    "probability": "0%",
                    "probability_float": 0.0,
                    "entry_zone": [0.0],
                    "take_profit": [0.0],
                    "stop_loss_atr": 0.0,
                    "trailing_step_atr": 0.0,
                    "position_sizing": {"kelly_fraction": 0.0, "recommended_margin_percentage": "0%", "suggested_leverage": "1x", "position_size": 0.0},
                    "atr": 0.0,
                    "timeframe": tf,
                    "reason": "No signal above threshold",
                    "winrate": "",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

    except Exception:
        logger.exception("[Quantuis] Error processing coin")


async def _coin_monitoring_loop(config: dict) -> None:
    symbols = config.get("data", {}).get("symbols", ["BTC/USDT:USDT"])
    interval_minutes = 15
    cycle_count = 0
    while True:
        if kill_switch_is_active():
            logger.warning("[Quantuis] Kill switch aktif. Menghentikan monitoring loop.")
            break

        cycle_count += 1
        try:
            logger.info(f"[Quantuis] Starting coin monitoring cycle {cycle_count} for {len(symbols)} symbols")
            for symbol in symbols:
                try:
                    await _process_coin(symbol, config)
                except Exception:
                    logger.exception(f"[Quantuis] Error processing {symbol} in monitoring cycle")
            logger.info(f"[Quantuis] Monitoring cycle {cycle_count} complete. Signals: {len(current_signals)}")
            _log_memory_usage("End monitoring cycle")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if cycle_count % 10 == 0:
                logger.info(f"[Quantuis] Explicit deep memory cleanup at cycle {cycle_count}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            logger.exception("[Quantuis] Error in monitoring cycle")
        await asyncio.sleep(interval_minutes * 60)


async def _continuous_learning_loop(config: dict) -> None:
    global feedback_engine
    feedback_engine = FeedbackLoopEngine(config=config)
    feedback_engine.start()
    while True:
        if kill_switch_is_active():
            logger.warning("[Quantuis] Kill switch aktif. Menghentikan continuous learning loop.")
            break

        try:
            await asyncio.sleep(300)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"[Quantuis] Error in continuous learning loop: {e}")


async def _position_monitoring_loop(config: dict) -> None:
    global shadow_trader
    interval_minutes = 1
    while True:
        if kill_switch_is_active():
            logger.warning("[Quantuis] Kill switch aktif. Menghentikan position monitoring loop.")
            break

        try:
            if shadow_trader is not None:
                shadow_trader.monitor_positions(price_fetcher=None)
        except Exception:
            logger.exception("[Quantuis] Error in position monitoring loop")
        await asyncio.sleep(interval_minutes * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global feedback_engine, monitoring_task, learning_task, shadow_trader, telegram_bot, celery_app
    
    config = load_config("config.yaml")
    set_seed(42)

    db_url = config.get("database", {}).get("url", "")
    if db_url:
        try:
            create_schema(override_url=db_url)
        except Exception as e:
            logger.error(f"[Quantuis] Failed to create/verify database schema: {e}")

    _load_models_from_disk("models")

    # Initialize Celery app for async inference
    global celery_app
    celery_app = get_celery_app(config)
    logger.info("[Quantuis] Celery app initialized for inference")

    feedback_engine = FeedbackLoopEngine(config=config)
    feedback_engine.start()

    global sentiment_engine
    sentiment_engine = SentimentEngine()

    global smart_money_tracker
    smart_money_tracker = SmartMoneyTracker()

    global self_tuning_engine
    self_tuning_engine = SelfTuningEngine(config=config)
    self_tuning_engine.start()
    logger.info("[Quantuis] Self-tuning scheduler started (Sunday 02:00 UTC)")

    shadow_trader = ShadowTrader()

    if kill_switch_is_active():
        logger.warning("[Quantuis] Kill switch aktif saat startup. Background tasks akan dihentikan segera.")

    monitoring_task = asyncio.create_task(_coin_monitoring_loop(config))
    logger.info("[Quantuis] Coin monitoring task started (15-min interval)")

    learning_task = asyncio.create_task(_continuous_learning_loop(config))
    logger.info("[Quantuis] Continuous learning task started")

    global position_monitor_task
    position_monitor_task = asyncio.create_task(_position_monitoring_loop(config))
    logger.info("[Quantuis] Position monitoring task started (1-min interval)")

    global app_start_time
    app_start_time = time.time()
    logger.info("[Quantuis] Quantuis API server started")

    telegram_bot_cfg = config.get("telegram_bot", {})
    if telegram_bot_cfg.get("enabled", False):
        try:
            from crypto_trading_framework.telegram_bot.bot import TelegramBot
            from crypto_trading_framework.telegram_bot.config import TelegramBotConfig

            telegram_bot_config = TelegramBotConfig(**telegram_bot_cfg)
            telegram_bot = TelegramBot(telegram_bot_config)
            asyncio.create_task(telegram_bot.start())
            logger.info("[Quantuis] Telegram bot started")
        except (OSError, RuntimeError) as e:
            logger.error(f"[Quantuis] Failed to start Telegram bot: {e}")
            telegram_bot = None
    else:
        telegram_bot = None

    yield

    for task_ref, name in (
        (monitoring_task, "monitoring"),
        (learning_task, "learning"),
        (position_monitor_task, "position_monitoring"),
    ):
        if task_ref is not None and not task_ref.done():
            task_ref.cancel()
            try:
                await task_ref
            except asyncio.CancelledError:
                pass

    if feedback_engine is not None:
        feedback_engine.stop()

    if self_tuning_engine is not None:
        self_tuning_engine.stop()

    if telegram_bot is not None:
        try:
            await telegram_bot.stop()
        except Exception:
            logger.exception("[Quantuis] Error stopping Telegram bot")

    logger.info("[Quantuis] Quantuis server shutting down")


app = FastAPI(title="Quantuis", version="5.0.0", lifespan=lifespan)


@app.get("/api/v1/signal")
async def get_signal(search: str | None = Query(None)) -> JSONResponse:
    if search is None or search.strip() == "":
        if not current_signals:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "pending",
                    "message": "Sinyal belum tersedia.\nQuantuis sedang menganalisis.\nHarap tunggu",
                },
            )
        return JSONResponse(status_code=200, content={"status": "ok", "signals": current_signals})

    search_upper = search.strip().upper()
    for symbol, signal_data in current_signals.items():
        symbol_upper = symbol.upper()
        if search_upper in symbol_upper:
            return JSONResponse(status_code=200, content={"status": "ok", "signal": signal_data})

    return JSONResponse(
        status_code=200,
        content={
            "status": "pending",
            "message": "Sinyal belum tersedia.\nQuantuis sedang menganalisis.\nHarap tunggu",
        },
    )


@app.get("/api/v1/performance")
async def get_performance() -> JSONResponse:
    if shadow_trader is None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "pending",
                "message": "Sinyal belum tersedia.\nQuantuis sedang menganalisis.\nHarap tunggu",
            },
        )
    try:
        data = shadow_trader.ledger.get_performance()
        return JSONResponse(status_code=200, content=data)
    except Exception as e:
        logger.error(f"[Quantuis] Error getting performance: {e}")
        return JSONResponse(
            status_code=200,
            content={
                "status": "pending",
                "message": "Sinyal belum tersedia.\nQuantuis sedang menganalisis.\nHarap tunggu",
            },
        )


@app.get("/api/v1/health")
async def get_health() -> JSONResponse:
    uptime = time.time() - app_start_time if app_start_time > 0 else 0
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "uptime_seconds": round(uptime, 2),
            "monitoring_task": monitoring_task is not None and not monitoring_task.done(),
            "learning_task": learning_task is not None and not learning_task.done(),
            "position_monitor_task": position_monitor_task is not None and not position_monitor_task.done(),
            "shadow_trader": shadow_trader is not None,
        },
    )


def _run_pipeline_sync(config: dict) -> None:
    symbol = config["data"]["symbols"][0]
    timeframes = config["data"]["timeframes"]
    lookback = config["data"]["lookback"]
    time_steps = config["ml"]["time_steps"]
    threshold = config["signal"]["threshold"]

    logger.info(f"[START] Mengambil data {symbol} untuk timeframe: {', '.join(timeframes)}")

    ingestion = DataIngestion(exchange_id=config["data"]["exchange_id"], symbol=symbol)
    try:
        raw_data = asyncio.run(ingestion.fetch_multi_timeframe(timeframes=timeframes, limit=lookback))
    except Exception as e:
        logger.error(f"Gagal mengambil data dari {config['data']['exchange_id']}: {e}")
        if config["data"]["fallback_enabled"]:
            logger.info("Mencoba yfinance sebagai fallback...")
            raw_data = {}
            interval_map = {"m5": "5m", "m15": "15m", "m30": "30m", "h1": "1h", "h4": "4h", "d1": "1d"}
            for tf in timeframes:
                df_yf = ingestion.fetch_yfinance(
                    ticker=config["data"]["yfinance_ticker"][0],
                    period=config["data"]["yfinance_period"],
                    interval=interval_map.get(tf, "1h"),
                )
                if df_yf is not None:
                    raw_data[tf] = df_yf

    if not raw_data:
        logger.critical("Tidak ada data yang berhasil diambil.")
        return

    for tf, df_raw in raw_data.items():
        logger.info(f"\n[PROCESS] Memproses timeframe: {tf}")
        df = add_all_indicators(df_raw.lazy())
        df = merge_market_data(df, {})
        df = df.collect()
        pipeline = MLPipeline(scaler_type=config["ml"]["scaler_type"])
        target_type = config.get("target", {}).get("type", "binary")
        df = pipeline.define_target(df, forward_periods=config["ml"]["forward_periods"], target_type=target_type)
        df = df.drop_nulls()

        if df.is_empty():
            continue

        features, _feature_cols = pipeline.prepare_features(df, feature_cols=config["ml"]["feature_cols"])
        targets = df.select("target").to_numpy().flatten()

        if len(features) <= time_steps:
            continue

        split_idx = int(len(features) * (1 - config["ml"]["test_size"]))
        features_train_raw = features[:split_idx]
        features_test_raw = features[max(0, split_idx - time_steps):]
        targets_train_raw = targets[:split_idx]
        targets_test_raw = targets[max(0, split_idx - time_steps):]

        scaled_train = pipeline.scale_features(features_train_raw, fit=True)
        scaled_test = pipeline.scale_features(features_test_raw, fit=False)

        if len(scaled_train) <= time_steps or len(scaled_test) <= time_steps:
            continue

        X_train, _y_train = pipeline.create_sequences(scaled_train, targets_train_raw, time_steps=time_steps)
        X_test, _y_test = pipeline.create_sequences(scaled_test, targets_test_raw, time_steps=time_steps)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        input_size = X_train.shape[2]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_type = config["model"]["type"]
        if model_type == "ensemble":
            model = create_model(model_type="lstm", input_size=input_size).to(device)
        else:
            model = create_model(model_type=model_type, input_size=input_size).to(device)

        # Run inference via Celery
        if celery_app is not None:
            task = celery_app.send_task(
                "inference.run",
                args=[
                    "models/best_weights.pth",
                    model_type if model_type != "ensemble" else "lstm",
                    input_size,
                    X_test.tolist(),
                    str(device),
                ],
                queue="inference",
            )
            try:
                preds_list = task.get(timeout=config.get("task_queue", {}).get("timeout", 120))
                preds = np.array(preds_list, dtype=np.float32)
            except Exception as e:
                logger.error(f"Inference task failed: {e}")
                continue
        else:
            model.eval()
            with torch.no_grad():
                x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                logits = model(x_test_tensor)
                preds = torch.sigmoid(logits).cpu().numpy().flatten()

        result_df = df.tail(len(preds)).with_columns(pl.Series("prob", preds))

        signal = generate_signal(
            result_df,
            threshold=threshold,
            current_price=df["close"].tail(1).item(),
            min_adx=config.get("signal", {}).get("min_adx", 25.0),
            require_volume_spike=config.get("signal", {}).get("require_volume_spike", False),
        )

        if signal:
            signal["symbol"] = symbol
            signal["timeframe"] = tf
            print_signal_table([signal])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "bot":
        config = load_config("config.yaml")
        bot = AutomatedTradingBot(config)
        bot.start()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)