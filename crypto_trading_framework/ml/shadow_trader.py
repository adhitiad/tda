"""
Shadow trading engine for virtual order simulation.
"""

from __future__ import annotations

from typing import Any

from crypto_trading_framework.core.logging import get_logger
from crypto_trading_framework.db.ledger import Ledger

logger = get_logger("shadow_trader")


class ShadowTrader:
    """
    Simulates trading execution with fees, leverage, and Kelly sizing.
    """

    TAKER_FEE_RATE = 0.0004  # 0.04%
    MAKER_FEE_RATE = 0.0002  # 0.02%

    def __init__(self, ledger: Ledger | None = None):
        self.ledger = ledger or Ledger()

    def calculate_fee(self, notional: float, is_taker: bool = True) -> float:
        rate = self.TAKER_FEE_RATE if is_taker else self.MAKER_FEE_RATE
        return abs(notional) * rate

    def execute_signal(self, signal: dict[str, Any]) -> int | None:
        if not signal or signal.get("status") != "success":
            return None

        direction = signal.get("direction", "LONG")
        entry_price = float(signal.get("entry", 0.0))
        if entry_price <= 0:
            return None

        wallet = self.ledger.get_wallet()
        current_balance = float(wallet.get("current_balance", 0.0))
        available_margin = float(wallet.get("available_margin", current_balance))

        position_sizing = signal.get("position_sizing", {})
        kelly_margin_pct = float(position_sizing.get("kelly_fraction", 0.0))
        suggested_leverage_str = position_sizing.get("suggested_leverage", "1x")
        leverage = float(suggested_leverage_str.replace("x", "")) if isinstance(suggested_leverage_str, str) else 1.0
        leverage = max(1.0, min(20.0, leverage))

        margin_amount = available_margin * kelly_margin_pct
        if margin_amount <= 0:
            logger.info("[ShadowTrader] Insufficient margin for signal")
            return None

        notional = margin_amount * leverage
        size = notional / entry_price
        open_fee = self.calculate_fee(notional, is_taker=True)

        if available_margin < margin_amount + open_fee:
            margin_amount = available_margin / (1.0 + self.TAKER_FEE_RATE)
            notional = margin_amount * leverage
            size = notional / entry_price
            open_fee = self.calculate_fee(notional, is_taker=True)

        side = "LONG" if direction == "LONG" else "SHORT"
        atr = float(signal.get("atr", 0.0))
        stop_loss_atr = float(signal.get("stop_loss", 0.0))
        take_profit = float(signal.get("take_profit", 0.0))
        trade_id = self.ledger.open_trade(
            symbol=signal.get("simbol", signal.get("symbol", "UNKNOWN")),
            side=side,
            entry_price=entry_price,
            size=size,
            leverage=f"{int(leverage)}x",
            fee=open_fee,
            atr=atr,
            stop_loss=stop_loss_atr,
            take_profit=take_profit,
        )

        new_available = available_margin - margin_amount - open_fee
        new_current = current_balance - open_fee
        self.ledger.update_balance(new_current, new_available)

        logger.info(f"[ShadowTrader] Executed trade {trade_id}: {side} size={size:.6f} leverage={int(leverage)}x fee={open_fee:.4f}")
        return trade_id

    def monitor_positions(self, price_fetcher: Any) -> None:
        open_trades = self.ledger.get_open_trades()
        if not open_trades:
            return

        for trade in open_trades:
            trade_id = trade.get("id")
            symbol = trade.get("symbol")
            side = trade.get("side")
            entry_price = float(trade.get("entry_price", 0.0))
            size = float(trade.get("size", 0.0))
            leverage_str = trade.get("leverage", "1x")
            leverage = float(leverage_str.replace("x", "")) if isinstance(leverage_str, str) else 1.0

            current_price = self._get_latest_price(price_fetcher, symbol)
            if current_price <= 0:
                continue

            close_trade = False
            reason = ""

            stored_atr = float(trade.get("atr", 0.0) or 0.0)
            stored_sl = float(trade.get("stop_loss", 0.0) or 0.0)
            stored_tp = float(trade.get("take_profit", 0.0) or 0.0)

            atr = stored_atr if stored_atr > 0 else (entry_price * 0.02)
            sl_multiplier = 1.5
            tp_multiplier = 2.0
            trailing_stop_atr_multiple = 1.0

            if side == "LONG":
                sl_price = stored_sl if stored_sl > 0 else (entry_price - (atr * sl_multiplier))
                tp_price = stored_tp if stored_tp > 0 else (entry_price + (atr * tp_multiplier))
                if current_price <= sl_price:
                    close_trade = True
                    reason = "Stop Loss hit"
                elif current_price >= tp_price:
                    close_trade = True
                    reason = "Take Profit hit"
                else:
                    potential_profit = current_price - entry_price
                    if potential_profit >= (atr * trailing_stop_atr_multiple):
                        new_sl = entry_price + (atr * 0.5)
                        if current_price - new_sl >= (atr * trailing_stop_atr_multiple):
                            pass
            else:
                sl_price = stored_sl if stored_sl > 0 else (entry_price + (atr * sl_multiplier))
                tp_price = stored_tp if stored_tp > 0 else (entry_price - (atr * tp_multiplier))
                if current_price >= sl_price:
                    close_trade = True
                    reason = "Stop Loss hit"
                elif current_price <= tp_price:
                    close_trade = True
                    reason = "Take Profit hit"
                else:
                    potential_profit = entry_price - current_price
                    if potential_profit >= (atr * trailing_stop_atr_multiple):
                        new_sl = entry_price - (atr * 0.5)
                        if new_sl - current_price >= (atr * trailing_stop_atr_multiple):
                            pass

            if close_trade:
                self._close_trade_with_calculations(trade_id, current_price, side, entry_price, size, leverage, reason)

    def _close_trade_with_calculations(
        self,
        trade_id: int,
        close_price: float,
        side: str,
        entry_price: float,
        size: float,
        leverage: float,
        reason: str,
    ) -> None:
        notional = size * close_price
        close_fee = self.calculate_fee(notional, is_taker=True)

        if side == "LONG":
            gross_pnl = (close_price - entry_price) * size * leverage
        else:
            gross_pnl = (entry_price - close_price) * size * leverage

        net_pnl = gross_pnl - close_fee
        wallet = self.ledger.get_wallet()
        current_balance = float(wallet.get("current_balance", 0.0))
        available_margin = float(wallet.get("available_margin", current_balance))

        entry_notional = size * entry_price
        margin_used = (entry_notional * leverage) / leverage if leverage > 0 else entry_notional
        new_available = available_margin + margin_used + net_pnl
        new_current = current_balance + net_pnl

        self.ledger.close_trade(
            trade_id=trade_id,
            close_price=close_price,
            pnl=net_pnl,
            fee=close_fee,
        )
        self.ledger.update_balance(new_current, new_available)

        logger.info(f"[ShadowTrader] Closed trade {trade_id}: {reason} PnL={net_pnl:.2f}")

    def _get_latest_price(self, price_fetcher: Any, symbol: str) -> float:
        try:
            if hasattr(price_fetcher, "get_latest_price"):
                return float(price_fetcher.get_latest_price(symbol) or 0.0)
            elif hasattr(price_fetcher, "fetch_ticker"):
                import asyncio
                ticker = asyncio.run(price_fetcher.fetch_ticker(symbol))
                return float(ticker.get("last", 0.0)) if ticker else 0.0
        except Exception as e:
            logger.error(f"[ShadowTrader] Error fetching price for {symbol}: {e}")
        return 0.0