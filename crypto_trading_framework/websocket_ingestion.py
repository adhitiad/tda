import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd
import polars as pl
import websockets

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("websocket")


@dataclass
class WebSocketBuffer:
    buffer: list[dict] = field(default_factory=list)
    flush_size: int = 100
    flush_interval: float = 5.0
    last_flush: float = field(default_factory=time.time)

    def add(self, candle: dict):
        self.buffer.append(candle)

    def should_flush(self) -> bool:
        return len(self.buffer) >= self.flush_size or (time.time() - self.last_flush >= self.flush_interval)

    def flush(self) -> pl.DataFrame | None:
        if not self.buffer:
            return None

        df = pl.DataFrame(self.buffer, orient="row")
        if "timestamp" in df.columns:
            df = df.with_columns(
                pl.col("timestamp").cast(pl.Int64).cast(pl.Datetime("ms"))
            )
        self.buffer.clear()
        self.last_flush = time.time()
        return df


class OKXWebSocketClient:
    TIMEFRAME_MAP: ClassVar[dict[str, str]] = {
        "m1": "candle1m",
        "m5": "candle5m",
        "m15": "candle15m",
        "m30": "candle30m",
        "h1": "candle1H",
        "h4": "candle4H",
        "d1": "candle1D",
    }

    def __init__(
        self,
        symbols: list[str],
        timeframes: list[str],
        buffer: WebSocketBuffer,
        on_flush=None,
    ):
        self.symbols = symbols
        self.timeframes = timeframes
        self.buffer = buffer
        self.on_flush = on_flush
        self.running = False
        self._ws = None

    @staticmethod
    def _ccxt_to_okx_inst_id(symbol: str) -> str:
        return symbol.replace("/", "-").replace(":USDT", "").replace(":USD", "")

    @staticmethod
    def _okx_inst_id_to_normalized(inst_id: str) -> str:
        return inst_id.replace("-", "/")

    def _build_subscription_message(self) -> str:
        args = []
        for symbol in self.symbols:
            inst_id = self._ccxt_to_okx_inst_id(symbol)
            for tf in self.timeframes:
                channel = self.TIMEFRAME_MAP.get(tf)
                if channel:
                    args.append({"channel": channel, "instId": inst_id})
        return json.dumps({"op": "subscribe", "args": args})

    @staticmethod
    def _parse_candle_message(data: list) -> dict | None:
        if not isinstance(data, list) or len(data) < 9:
            return None
        ts, o, h, l, c, vol = data[0], data[1], data[2], data[3], data[4], data[5]
        try:
            return {
                "timestamp": int(ts),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(vol),
            }
        except (ValueError, TypeError):
            return None

    async def _handle_message(self, message: str):
        try:
            msg = json.loads(message)
            if msg.get("event") == "subscribe":
                logger.info(f"[OKX WS] Subscribed: {msg}")
                return
            if msg.get("event") == "error":
                logger.error(f"[OKX WS] Error: {msg}")
                return

            arg = msg.get("arg", {})
            inst_id = arg.get("instId", "")
            channel = arg.get("channel", "")
            normalized_symbol = self._okx_inst_id_to_normalized(inst_id)

            timeframe = ""
            for tf, ch in self.TIMEFRAME_MAP.items():
                if ch == channel:
                    timeframe = tf
                    break

            data = msg.get("data")
            if not data:
                return

            for candle_data in data:
                candle = self._parse_candle_message(candle_data)
                if candle:
                    candle["symbol"] = normalized_symbol
                    candle["timeframe"] = timeframe
                    self.buffer.add(candle)

            if self.buffer.should_flush():
                df = self.buffer.flush()
                if df is not None and self.on_flush:
                    await self.on_flush(df)
        except Exception as e:
            logger.error(f"[OKX WS] Gagal parse message: {e}")

    async def _connect_and_listen(self):
        url = "wss://ws.okx.com:8443/ws/v5/public"
        sub_msg = self._build_subscription_message()
        backoff = 1
        max_backoff = 60

        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    self._ws = ws
                    logger.info("[OKX WS] Terhubung ke OKX WebSocket")
                    await ws.send(sub_msg)
                    backoff = 1

                    async for message in ws:
                        if not self.running:
                            break
                        await self._handle_message(message)
            except Exception as e:
                logger.error(f"[OKX WS] Koneksi terputus: {e}. Reconnect dalam {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def start(self):
        self.running = True
        await self._connect_and_listen()

    async def stop(self):
        self.running = False
        if self._ws:
            await self._ws.close()


async def _flush_to_db_and_cache(df: pl.DataFrame, source: str = "websocket"):
    if df is None or df.is_empty():
        return

    try:
        from crypto_trading_framework.db.database import (
            update_ingestion_state,
            upsert_ohlcv_dataframe,
        )
        from crypto_trading_framework.db.redis_cache import get_redis_cache

        pdf = df.to_pandas()
        symbol_cols = pdf["symbol"].unique() if "symbol" in pdf.columns else []
        timeframe_cols = pdf["timeframe"].unique() if "timeframe" in pdf.columns else []

        if symbol_cols.size > 0 and timeframe_cols.size > 0:
            for symbol in symbol_cols:
                for timeframe in timeframe_cols:
                    subset = pdf[
                        (pdf["symbol"] == symbol) & (pdf["timeframe"] == timeframe)
                    ]
                    if subset.empty:
                        continue
                    upsert_ohlcv_dataframe(subset, source=source)
                    last_ts = subset["timestamp"].max()
                    if pd.notna(last_ts):
                        update_ingestion_state(
                            symbol=symbol,
                            timeframe=timeframe,
                            source=source,
                            last_ingested_at=pd.Timestamp(last_ts).to_pydatetime(),
                            row_count=len(subset),
                        )

                    cache = get_redis_cache()
                    cache.cache_ohlcv(symbol, timeframe, subset, ttl=60)

        logger.info(f"[WS Buffer] Flushed {len(pdf)} rows ke DB dan cache")
    except Exception as e:
        logger.error(f"[WS Buffer] Gagal flush: {e}")


async def start_websocket_ingestion(config: dict | None = None):
    if config is None:
        try:
            from main import load_config
            config = load_config()
        except Exception:
            config = {}

    ws_cfg = config.get("websocket", {})
    if not ws_cfg.get("enabled", False):
        logger.info("[WS] WebSocket ingestion dinonaktifkan")
        return

    symbols = config.get("data", {}).get("symbols", [])[: config.get("trading", {}).get("max_symbols", 10)]
    timeframes = config.get("data", {}).get("timeframes", ["m15", "h1", "h4", "d1"])

    buffer = WebSocketBuffer(
        flush_size=ws_cfg.get("buffer_size", 100),
        flush_interval=ws_cfg.get("buffer_interval", 5.0),
    )

    client = OKXWebSocketClient(
        symbols=symbols,
        timeframes=timeframes,
        buffer=buffer,
        on_flush=_flush_to_db_and_cache,
    )

    logger.info(f"[WS] Memulai WebSocket ingestion untuk {len(symbols)} simbol")
    await client.start()
