"""
Shadow trading ledger using SQLAlchemy and TimescaleDB/PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from crypto_trading_framework.core.logging import get_logger
from crypto_trading_framework.db.database import session_scope, Wallet, TradeHistory

logger = get_logger("ledger")


class Ledger:
    """Manages virtual wallet and trade history via SQLAlchemy."""

    def __init__(self) -> None:
        pass

    def _init_wallet_if_empty(self) -> None:
        with session_scope() as session:
            if session.query(Wallet).filter_by(id=1).first() is None:
                now = datetime.now(timezone.utc)
                wallet = Wallet(
                    id=1,
                    initial_balance=10000.0,
                    current_balance=10000.0,
                    available_margin=10000.0,
                    updated_at=now,
                )
                session.add(wallet)
                logger.info("[Ledger] Wallet initialized with $10,000")

    def get_wallet(self) -> dict[str, Any]:
        with session_scope() as session:
            wallet = session.query(Wallet).filter_by(id=1).first()
            if wallet is None:
                return {}
            return {
                "id": wallet.id,
                "initial_balance": wallet.initial_balance,
                "current_balance": wallet.current_balance,
                "available_margin": wallet.available_margin,
                "updated_at": wallet.updated_at.isoformat() if wallet.updated_at else None,
            }

    def update_balance(self, new_current_balance: float, new_available_margin: float | None = None) -> None:
        now = datetime.now(timezone.utc)
        if new_available_margin is None:
            new_available_margin = new_current_balance
        with session_scope() as session:
            wallet = session.query(Wallet).filter_by(id=1).first()
            if wallet is not None:
                wallet.current_balance = new_current_balance
                wallet.available_margin = new_available_margin
                wallet.updated_at = now

    def open_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        leverage: str,
        fee: float,
        atr: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> int:
        now = datetime.now(timezone.utc)
        with session_scope() as session:
            trade = TradeHistory(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                size=size,
                leverage=leverage,
                fee=fee,
                status="OPEN",
                opened_at=now,
                atr=atr,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            session.add(trade)
            session.flush()
            trade_id = trade.id
        logger.info(f"[Ledger] Opened trade {trade_id} for {symbol} {side} size={size} leverage={leverage}")
        return trade_id

    def close_trade(
        self,
        trade_id: int,
        close_price: float,
        pnl: float,
        fee: float,
        status: str = "CLOSED",
    ) -> None:
        now = datetime.now(timezone.utc)
        with session_scope() as session:
            trade = session.query(TradeHistory).filter_by(id=trade_id).first()
            if trade is not None:
                trade.close_price = close_price
                trade.pnl = pnl
                trade.fee = fee
                trade.status = status
                trade.closed_at = now
        logger.info(f"[Ledger] Closed trade {trade_id} with PnL={pnl:.2f}")

    def get_open_trades(self) -> list[dict[str, Any]]:
        with session_scope() as session:
            trades = session.query(TradeHistory).filter_by(status="OPEN").all()
            return [self._trade_to_dict(trade) for trade in trades]

    def get_closed_trades(self) -> list[dict[str, Any]]:
        with session_scope() as session:
            trades = session.query(TradeHistory).filter_by(status="CLOSED").all()
            return [self._trade_to_dict(trade) for trade in trades]

    def get_performance(self) -> dict[str, Any]:
        with session_scope() as session:
            closed_trades = session.query(TradeHistory).filter_by(status="CLOSED").all()
            open_trades = session.query(TradeHistory).filter_by(status="OPEN").all()
            wallet = session.query(Wallet).filter_by(id=1).first()

            closed = [self._trade_to_dict(trade) for trade in closed_trades]
            open_trades_data = [self._trade_to_dict(trade) for trade in open_trades]

            initial = wallet.initial_balance if wallet else 0.0
            current = wallet.current_balance if wallet else 0.0
            total_net_profit = current - initial
            roi = ((current - initial) / initial * 100.0) if initial > 0 else 0.0

            total_trades = len(closed)
            winning = sum(1 for t in closed if t.get("pnl", 0.0) > 0)
            losing = total_trades - winning
            win_rate = (winning / total_trades * 100.0) if total_trades > 0 else 0.0

            return {
                "initial_balance": round(initial, 2),
                "current_balance": round(current, 2),
                "total_net_profit": round(total_net_profit, 2),
                "roi_percentage": f"{roi:.2f}%",
                "win_rate": f"{win_rate:.2f}%",
                "total_trades": total_trades,
                "winning_trades": winning,
                "losing_trades": losing,
                "open_positions": open_trades_data,
            }

    @staticmethod
    def _trade_to_dict(trade: TradeHistory) -> dict[str, Any]:
        return {
            "id": trade.id,
            "symbol": trade.symbol,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "close_price": trade.close_price,
            "size": trade.size,
            "leverage": trade.leverage,
            "pnl": trade.pnl,
            "fee": trade.fee,
            "status": trade.status,
            "opened_at": trade.opened_at.isoformat() if trade.opened_at else None,
            "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
            "atr": trade.atr,
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
        }