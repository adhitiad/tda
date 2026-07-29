import asyncio
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from crypto_trading_framework.core.alerting import alert_execution_error
from crypto_trading_framework.core.bot import AutomatedTradingBot
from crypto_trading_framework.core.logging import get_logger

logger = get_logger("api")


class BotController:
    _instance = None

    def __init__(self):
        self.bot: AutomatedTradingBot | None = None
        self.config: dict | None = None
        self.task: asyncio.Task | None = None
        self.started_at: str | None = None
        self.status = "stopped"

    @classmethod
    def get_instance(cls) -> "BotController":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure(self, config: dict):
        self.config = config
        self.bot = AutomatedTradingBot(config)

    async def start(self):
        if self.status == "running":
            return {"status": "already_running"}
        if self.bot is None:
            raise RuntimeError("Bot belum dikonfigurasi")
        self.task = asyncio.create_task(self.bot.start_async())
        self.status = "running"
        self.started_at = datetime.now().isoformat()
        return {"status": "started", "started_at": self.started_at}

    async def stop(self):
        if self.status != "running":
            return {"status": "not_running"}
        if self.bot:
            self.bot.stop()
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.status = "stopped"
        self.started_at = None
        return {"status": "stopped"}

    def get_status(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "started_at": self.started_at,
            "symbols": self.bot.symbols if self.bot else [],
            "dry_run": self.bot.dry_run if self.bot else None,
            "trained_models": list(self.bot.trained_models.keys()) if self.bot else [],
        }


def create_app(config: dict | None = None) -> FastAPI:
    api_cfg = config.get("api", {}) if config else {}
    app = FastAPI(title="Trading Bot API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_cfg.get("allowed_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    controller = BotController.get_instance()
    if config:
        controller.configure(config)

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        api_key = api_cfg.get("api_key", "")
        if api_key:
            header_key = request.headers.get("X-API-Key")
            if header_key != api_key:
                return JSONResponse(status_code=403, content={"detail": "Invalid API key"})
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    @app.get("/control/status")
    async def get_status():
        return controller.get_status()

    @app.post("/control/start")
    async def start_bot():
        try:
            return await controller.start()
        except Exception as e:
            logger.error(f"Gagal start bot via API: {e}")
            alert_execution_error("api", str(e), {"endpoint": "/control/start"})
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/control/stop")
    async def stop_bot():
        try:
            return await controller.stop()
        except Exception as e:
            logger.error(f"Gagal stop bot via API: {e}")
            alert_execution_error("api", str(e), {"endpoint": "/control/stop"})
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/signals")
    async def get_signals():
        if controller.bot is None:
            return {"signals": []}
        signals = []
        for symbol, info in controller.bot.trained_models.items():
            signals.append({
                "symbol": symbol,
                "model_loaded": True,
                "time_steps": info.get("time_steps"),
                "feature_cols": info.get("feature_cols", []),
            })
        return {"signals": signals}

    @app.get("/models")
    async def get_models():
        if controller.bot is None:
            return {"models": []}
        models = []
        for symbol, info in controller.bot.trained_models.items():
            models.append({
                "symbol": symbol,
                "model_type": type(info.get("model")).__name__,
                "time_steps": info.get("time_steps"),
                "feature_cols": info.get("feature_cols", []),
            })
        return {"models": models}

    @app.post("/backtest")
    async def trigger_backtest(payload: dict[str, Any]):
        if controller.bot is None:
            raise HTTPException(status_code=400, detail="Bot belum dikonfigurasi")
        symbol = payload.get("symbol")
        timeframe = payload.get("timeframe", "1h")
        if not symbol:
            raise HTTPException(status_code=422, detail="symbol diperlukan")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "queued",
            "message": "Backtest akan dijalankan pada siklus pelatihan berikutnya",
        }

    return app
