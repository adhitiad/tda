import asyncio
import time
from datetime import datetime
from decimal import Decimal
from typing import Optional

import ccxt.async_support as ccxt_async

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("executor")


class OrderExecutor:
    def __init__(self, exchange_id: str, symbol: str, dry_run: bool = True, capital: float = 10000.0, risk_per_trade: float = 0.02):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.dry_run = dry_run
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.exchange = getattr(ccxt_async, exchange_id)()
        self.active_trades = {}

    async def execute_signal(self, signal: dict) -> Optional[dict]:
        direction = signal.get("direction", "LONG")
        action = signal.get("action", "BUY")
        entry = signal.get("entry")
        tp = signal.get("take_profit")
        sl = signal.get("stop_loss")

        if not entry or not tp or not sl:
            logger.error("[EXECUTOR] Sinyal tidak memiliki entry/TP/SL")
            return None

        side = "buy" if action == "BUY" else "sell"
        order_type = "limit"
        amount = self._calculate_position_size(entry, sl)

        if self.dry_run:
            logger.info(f"[EXECUTOR][DRY RUN] {side.upper()} {self.symbol} @ {entry}, TP={tp}, SL={sl}, amount={amount}")
            trade = {
                "id": f"dry_{int(time.time()*1000)}",
                "symbol": self.symbol,
                "side": side,
                "type": order_type,
                "price": entry,
                "amount": amount,
                "tp": tp,
                "sl": sl,
                "status": "dry_run",
                "timestamp": datetime.now().isoformat(),
            }
            self.active_trades[trade["id"]] = trade
            return trade

        try:
            order = await self.exchange.create_order(self.symbol, order_type, side, amount, entry)
            logger.info(f"[EXECUTOR] Order executed: {order['id']} {side} {amount} {self.symbol} @ {entry}")
            trade = {
                "id": order["id"],
                "symbol": self.symbol,
                "side": side,
                "type": order_type,
                "price": entry,
                "amount": amount,
                "tp": tp,
                "sl": sl,
                "status": "open",
                "timestamp": datetime.now().isoformat(),
            }
            self.active_trades[trade["id"]] = trade
            return trade
        except Exception as e:
            logger.error(f"[EXECUTOR] Gagal eksekusi order: {e}")
            return None

    def _calculate_position_size(self, entry: float, sl: float) -> float:
        risk_amount = self.capital * self.risk_per_trade
        risk_per_unit = abs(entry - sl)
        if risk_per_unit == 0:
            return 0.0
        amount = risk_amount / risk_per_unit
        return round(amount, 6)

    async def close_all(self):
        for trade_id, trade in list(self.active_trades.items()):
            try:
                if trade["status"] == "open" and not self.dry_run:
                    side = "sell" if trade["side"] == "buy" else "buy"
                    await self.exchange.create_order(self.symbol, "market", side, trade["amount"])
                logger.info(f"[EXECUTOR] Trade {trade_id} closed")
            except Exception as e:
                logger.error(f"[EXECUTOR] Gagal close trade {trade_id}: {e}")
        self.active_trades.clear()

    async def close(self):
        await self.close_all()
        await self.exchange.close()
