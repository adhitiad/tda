from crypto_trading_framework.cli import main as cli_main

try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Quantuis", version="5.0.0")

app_start_time = 0.0


@app.get("/api/v1/signal")
async def get_signal() -> JSONResponse:
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
        },
    )


@app.on_event("startup")
async def startup() -> None:
    global app_start_time
    app_start_time = time.time()


if __name__ == "__main__":
    cli_main()