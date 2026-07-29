import asyncio

import numpy as np
import polars as pl
import torch

from crypto_trading_framework.core.alerting import (
    AlertManager,
    alert_drawdown_breach,
    alert_execution_error,
    alert_training_failure,
)
from crypto_trading_framework.core.checkpoint_utils import save_checkpoint
from crypto_trading_framework.core.logging import get_logger
from crypto_trading_framework.data_ingestion import DataIngestion
from crypto_trading_framework.ml.drift_detector import DriftConfig, DriftDetector
from crypto_trading_framework.core.executor import OrderExecutor
from crypto_trading_framework.ml.feature_store import (
    compute_and_store_features,
    get_latest_features,
)
from crypto_trading_framework.core.indicators import add_all_indicators
from crypto_trading_framework.ml.ml_pipeline import MLPipeline
from crypto_trading_framework.ml.model import create_model
from crypto_trading_framework.ml.model_registry import ModelRegistry, ModelRegistryConfig
from crypto_trading_framework.portfolio import PortfolioRiskManager, PortfolioRiskConfig
from crypto_trading_framework.ml.rule_signals import generate_rule_based_signal
from crypto_trading_framework.core.scheduler import TrainingScheduler
from crypto_trading_framework.ml.signals import generate_signal
from crypto_trading_framework.ml.task_queue import BoundedTaskPool, TaskQueueConfig

logger = get_logger("bot")


class AutomatedTradingBot:
    def __init__(self, config: dict):
        self.config = config
        self.symbols = config["data"]["symbols"][: config["trading"]["max_symbols"]]
        self.dry_run = config["trading"]["dry_run"]
        self.running = False
        self.scheduler: TrainingScheduler | None = None
        self.executors = {}
        self.trained_models = {}

        drift_cfg = config.get("drift_detection", {})
        self.drift_detector = DriftDetector(DriftConfig(**drift_cfg)) if drift_cfg.get("enabled", True) else None

        reg_cfg = config.get("model_registry", {})
        self.model_registry = ModelRegistry(ModelRegistryConfig(**reg_cfg)) if reg_cfg.get("enabled", True) else None

        tq_cfg = config.get("task_queue", {})
        self.task_pool = BoundedTaskPool(TaskQueueConfig(**tq_cfg)) if tq_cfg.get("enabled", True) else None

        pr_cfg = config.get("portfolio_risk", {})
        self.portfolio = PortfolioRiskManager(PortfolioRiskConfig(**pr_cfg)) if pr_cfg.get("enabled", True) else None

        alerting_cfg = config.get("alerting", {})
        if alerting_cfg.get("enabled", False):
            AlertManager.get_instance().configure(alerting_cfg)

    def _training_callback(self):
        logger.info("[BOT] Memulai siklus pelatihan otomatis")
        asyncio.run(self._train_all_symbols())
        logger.info("[BOT] Siklus pelatihan selesai")

    def _stop_callback(self):
        logger.info("[BOT] Bot akan berhenti setelah pelatihan selesai")
        self.running = False

    async def _train_single_symbol(self, symbol: str):
        logger.info(f"[BOT] Melatih model untuk {symbol}")
        ingestion = DataIngestion(
            exchange_id=self.config["data"]["exchange_id"],
            symbol=symbol,
        )
        try:
            raw_data = await ingestion.fetch_multi_timeframe(
                timeframes=self.config["data"]["timeframes"],
                limit=self.config["data"]["lookback"],
            )
        except Exception as e:
            logger.error(f"[BOT] Gagal fetch {symbol}: {e}")
            alert_training_failure(symbol, str(e), {"stage": "fetch"})
            await ingestion.close()
            return
        await ingestion.close()

        if not raw_data:
            logger.warning(f"[BOT] Tidak ada data untuk {symbol}")
            return

        primary_tf = self.config["data"]["timeframes"][0]
        if primary_tf not in raw_data:
            return

        df = add_all_indicators(raw_data[primary_tf])
        pipeline = MLPipeline()
        df = pipeline.define_target(df, forward_periods=self.config["ml"]["forward_periods"])
        df = df.drop_nulls()

        if df.is_empty():
            return

        fs_cfg = self.config.get("feature_store", {})
        if fs_cfg.get("compute_on_ingest", True):
            try:
                compute_and_store_features(raw_data[primary_tf].to_pandas(), symbol, primary_tf)
            except Exception as e:
                logger.warning(f"[BOT] Gagal simpan features ke store: {e}")

        features, feature_cols = pipeline.prepare_features(df, feature_cols=self.config["ml"]["feature_cols"])
        targets = df.select("target").to_numpy().flatten()
        scaled = pipeline.scale_features(features, fit=True)
        X, y_seq = pipeline.create_sequences(scaled, targets, time_steps=self.config["ml"]["time_steps"])

        if len(X) <= self.config["ml"]["time_steps"]:
            return

        X_train, X_test, y_train, y_test = pipeline.train_test_split_sequences(
            X, y_seq, test_size=self.config["ml"]["test_size"]
        )
        input_size = X_train.shape[2]
        device = "cpu"

        try:
            from crypto_trading_framework.ml.training import train_model
            model_type = self.config["model"]["type"]

            if model_type == "ensemble":
                from crypto_trading_framework.ml.ensemble import EnsembleModel
                model = EnsembleModel(input_size=input_size, device=torch.device("cpu"))
                model.fit(X_train, y_train, X_test, y_test)
            else:
                model = create_model(
                    model_type=model_type,
                    input_size=input_size,
                    hidden_size=self.config["model"]["hidden_size"],
                    num_layers=self.config["model"]["num_layers"],
                    dropout=self.config["model"]["dropout"],
                )
                model, _ = train_model(
                    X_train, y_train, model,
                    self.config["model"]["epochs"],
                    self.config["model"]["learning_rate"],
                    device,
                )
        except Exception as e:
            logger.error(f"[BOT] Training gagal {symbol}: {e}")
            alert_training_failure(symbol, str(e), {"stage": "training"})
            return

        self.trained_models[symbol] = {
            "model": model,
            "scaler": pipeline.scaler,
            "feature_cols": feature_cols,
            "time_steps": self.config["ml"]["time_steps"],
        }

        if self.drift_detector is not None:
            try:
                model.eval()
                with torch.no_grad():
                    x_tensor = torch.tensor(X_test, dtype=torch.float32)
                    logits = model(x_tensor)
                    val_preds = torch.sigmoid(logits).numpy().flatten()
                self.drift_detector.set_baseline(symbol, val_preds, X_test.mean(axis=1), feature_cols)
            except Exception as e:
                logger.warning(f"[BOT] Gagal set drift baseline {symbol}: {e}")

        if self.model_registry is not None:
            try:
                model_path, scaler_path, meta_path = save_checkpoint(
                    model, pipeline.scaler,
                    {
                        "params": self.config["model"],
                        "accuracy": None,
                        "loss": None,
                        "symbol": symbol,
                        "timeframe": primary_tf,
                    },
                    self.config["model"]["checkpoint_dir"],
                    primary_tf,
                    is_best=True,
                )
                self.model_registry.register(
                    model_type=self.config["model"]["type"],
                    symbol=symbol,
                    timeframe=primary_tf,
                    model_path=model_path,
                    scaler_path=scaler_path,
                    meta_path=meta_path,
                    hyperparameters=self.config["model"],
                    tags=["auto"],
                )
            except Exception as e:
                logger.warning(f"[BOT] Gagal register model {symbol}: {e}")

        logger.info(f"[BOT] Model {symbol} berhasil dilatih")

    async def _train_all_symbols(self):
        if self.task_pool is not None:
            await self.task_pool.map(self._train_single_symbol, self.symbols)
        else:
            for symbol in self.symbols:
                await self._train_single_symbol(symbol)

    async def _monitor_and_trade(self):
        logger.info("[BOT] Memulai monitoring dan trading")
        initial_capital = float(self.config["backtest"].get("initial_capital", 10000.0))
        if self.portfolio is not None:
            self.portfolio.update_capital(initial_capital)

        while self.running:
            for symbol in self.symbols:
                if symbol not in self.trained_models:
                    continue
                try:
                    if self.portfolio is not None:
                        breached, drawdown = self.portfolio.check_portfolio_drawdown()
                        if breached:
                            logger.warning(f"[BOT] Portfolio drawdown breach: {drawdown:.2%}. Pausing new trades.")
                            continue

                    signal = await self._get_signal(symbol)
                    if signal:
                        executor = OrderExecutor(
                            exchange_id=self.config["data"]["exchange_id"],
                            symbol=symbol,
                            dry_run=self.dry_run,
                        )
                        trade = await executor.execute_signal(signal)
                        await executor.close()

                        if trade and not self.dry_run and trade.get("status") == "open":
                            pnl = float(trade.get("pnl", 0.0))
                            current_capital = self.portfolio.capital if self.portfolio else initial_capital
                            current_capital += pnl
                            if self.portfolio is not None:
                                self.portfolio.update_capital(current_capital)
                            peak_capital = self.portfolio.peak_capital if self.portfolio else max(initial_capital, current_capital)
                            drawdown = (peak_capital - current_capital) / peak_capital if peak_capital > 0 else 0.0
                            max_drawdown_pct = float(self.config["backtest"].get("max_drawdown_pct", 0.20))
                            if drawdown >= max_drawdown_pct:
                                alert_drawdown_breach(symbol, drawdown, max_drawdown_pct, {
                                    "capital": f"{current_capital:.2f}",
                                    "peak": f"{peak_capital:.2f}",
                                })
                except Exception as e:
                    logger.error(f"[BOT] Error monitoring {symbol}: {e}")
                    alert_execution_error(symbol, str(e), {"stage": "monitoring"})
            await asyncio.sleep(60)

    async def _get_signal(self, symbol: str) -> dict | None:
        primary_tf = self.config["data"]["timeframes"][0]
        model_info = self.trained_models[symbol]
        pipeline = MLPipeline()
        pipeline.scaler = model_info["scaler"]

        fs_cfg = self.config.get("feature_store", {})
        use_feature_store = fs_cfg.get("enabled", True)

        if use_feature_store:
            features_df = get_latest_features(symbol, primary_tf, n_rows=model_info["time_steps"] + 50)
            if features_df is not None and not features_df.empty:
                df = pl.from_pandas(features_df)
                df = pipeline.define_target(df, forward_periods=self.config["ml"]["forward_periods"])
                df = df.drop_nulls()

                if not df.is_empty():
                    features, _ = pipeline.prepare_features(df, feature_cols=model_info["feature_cols"])
                    scaled = pipeline.scale_features(features, fit=False)
                    X, _ = pipeline.create_sequences(scaled, df.select("target").to_numpy().flatten(), time_steps=model_info["time_steps"])

                    if len(X) > 0:
                        model = model_info["model"]
                        model.eval()
                        import torch
                        with torch.no_grad():
                            x_tensor = torch.tensor(X[-1:], dtype=torch.float32)
                            logits = model(x_tensor)
                            prob = torch.sigmoid(logits).item()

                        result_df = df.tail(len(X)).with_columns(pl.Series("prob", [prob] * len(X)))
                        signal = generate_signal(
                            result_df,
                            threshold=self.config["signal"]["threshold"],
                            current_price=df["close"].tail(1).item(),
                            min_adx=self.config["signal"].get("min_adx", 25.0),
                            require_volume_spike=self.config["signal"].get("require_volume_spike", False),
                        )

                        if self.drift_detector is not None:
                            feature_vec = scaled[-1] if len(scaled) > 0 else np.zeros(len(model_info["feature_cols"]))
                            self.drift_detector.update(symbol, prob, feature_vec)

                        if signal is None and self.config["signal"].get("use_rule_fallback", False):
                            signal = generate_rule_based_signal(df, min_confluences=self.config["signal"].get("min_confluences", 3))

                        return signal

        ingestion = DataIngestion(
            exchange_id=self.config["data"]["exchange_id"],
            symbol=symbol,
        )
        try:
            raw_data = await ingestion.fetch_multi_timeframe(
                timeframes=self.config["data"]["timeframes"],
                limit=self.config["data"]["lookback"],
            )
        except Exception:
            return None
        finally:
            await ingestion.close()

        if primary_tf not in raw_data:
            return None

        df = add_all_indicators(raw_data[primary_tf])
        df = pipeline.define_target(df, forward_periods=self.config["ml"]["forward_periods"])
        df = df.drop_nulls()

        if df.is_empty():
            return None

        features, _ = pipeline.prepare_features(df, feature_cols=model_info["feature_cols"])
        scaled = pipeline.scale_features(features, fit=False)
        X, _ = pipeline.create_sequences(scaled, df.select("target").to_numpy().flatten(), time_steps=model_info["time_steps"])

        if len(X) == 0:
            return None

        model = model_info["model"]
        model.eval()
        import torch
        with torch.no_grad():
            x_tensor = torch.tensor(X[-1:], dtype=torch.float32)
            logits = model(x_tensor)
            prob = torch.sigmoid(logits).item()

        result_df = df.tail(len(X)).with_columns(pl.Series("prob", [prob] * len(X)))
        signal = generate_signal(
            result_df,
            threshold=self.config["signal"]["threshold"],
            current_price=df["close"].tail(1).item(),
            min_adx=self.config["signal"].get("min_adx", 25.0),
            require_volume_spike=self.config["signal"].get("require_volume_spike", False),
        )

        if self.drift_detector is not None:
            feature_vec = scaled[-1] if len(scaled) > 0 else np.zeros(len(model_info["feature_cols"]))
            self.drift_detector.update(symbol, prob, feature_vec)

        if signal is None and self.config["signal"].get("use_rule_fallback", False):
            signal = generate_rule_based_signal(df, min_confluences=self.config["signal"].get("min_confluences", 3))

        return signal

    def start(self):
        if self.running:
            logger.warning("[BOT] Bot sudah berjalan")
            return
        self.running = True
        self.scheduler = TrainingScheduler(
            training_callback=self._training_callback,
            stop_callback=self._stop_callback,
        )
        self.scheduler.start()
        next_training = self.scheduler.get_next_training_time()
        logger.info(f"[BOT] Bot dimulai. Pelatihan berikutnya: {next_training}")
        try:
            asyncio.run(self._monitor_and_trade())
        except KeyboardInterrupt:
            logger.info("[BOT] Bot dihentikan manual")
        finally:
            self.stop()

    async def start_async(self):
        if self.running:
            logger.warning("[BOT] Bot sudah berjalan")
            return
        self.running = True
        self.scheduler = TrainingScheduler(
            training_callback=self._training_callback,
            stop_callback=self._stop_callback,
        )
        self.scheduler.start()
        next_training = self.scheduler.get_next_training_time()
        logger.info(f"[BOT] Bot dimulai (async). Pelatihan berikutnya: {next_training}")
        await self._monitor_and_trade()

    def stop(self):
        self.running = False
        if self.scheduler:
            self.scheduler.stop()
        logger.info("[BOT] Bot dihentikan")
