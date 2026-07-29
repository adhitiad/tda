from dataclasses import dataclass

import numpy as np
import polars as pl

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("backtest")


@dataclass
class TradeResult:
    """Hasil dari satu trade pada backtesting."""
    entry_time: str
    exit_time: str
    direction: str
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float
    size: float
    pnl: float
    pnl_pct: float
    result: str  # 'win', 'loss', 'open'
    fee_paid: float = 0.0
    slippage_paid: float = 0.0


@dataclass
class Position:
    """Representasi posisi trading aktif."""
    direction: str
    entry_price: float
    entry_time: str
    entry_idx: int
    tp_price: float
    sl_price: float
    size: float
    atr: float
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0


class Backtester:
    """Engine backtesting untuk evaluasi strategi trading dengan risk management."""

    def __init__(
        self,
        tp_pct: float = 0.03,
        sl_pct: float = 0.015,
        initial_capital: float = 10000.0,
        position_size_method: str = "fixed",
        atr_multiplier: float = 1.0,
        max_risk_per_trade: float = 0.02,
        trailing_stop_enabled: bool = False,
        trailing_stop_activation_pct: float = 0.015,
        trailing_stop_distance_pct: float = 0.01,
        max_drawdown_pct: float = 0.20,
        transaction_fee_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        forward_periods: int = 20,
    ):
        """
        Inisialisasi Backtester.

        :param tp_pct: Take Profit percentage
        :param sl_pct: Stop Loss percentage
        :param initial_capital: Modal awal
        :param position_size_method: 'fixed' atau 'atr'
        :param atr_multiplier: Multiplier ATR untuk position sizing
        :param max_risk_per_trade: Maksimum risiko per trade (persentase modal)
        :param trailing_stop_enabled: Aktifkan trailing stop
        :param trailing_stop_activation_pct: Profit threshold untuk mengaktifkan trailing stop
        :param trailing_stop_distance_pct: Jarak trailing stop dari harga saat ini
        :param max_drawdown_pct: Circuit breaker max drawdown
        :param transaction_fee_pct: Biaya transaksi (fee)
        :param slippage_pct: Slippage per trade
        :param forward_periods: Maksimum candle untuk menutup posisi
        """
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.initial_capital = initial_capital
        self.position_size_method = position_size_method
        self.atr_multiplier = atr_multiplier
        self.max_risk_per_trade = max_risk_per_trade
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_activation_pct = trailing_stop_activation_pct
        self.trailing_stop_distance_pct = trailing_stop_distance_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.transaction_fee_pct = transaction_fee_pct
        self.slippage_pct = slippage_pct
        self.forward_periods = forward_periods

    def run_backtest(
        self,
        df: pl.DataFrame,
        prob_col: str = "prob",
        threshold: float = 0.75,
    ) -> dict:
        """
        Menjalankan backtest pada data historis dengan risk management lengkap.

        :param df: Polars DataFrame dengan kolom close, high, low, prob, atr
        :param prob_col: Nama kolom probabilitas
        :param threshold: Ambang batas probabilitas untuk entry
        :return: Dictionary hasil backtest
        """
        trades: list[TradeResult] = []
        capital = self.initial_capital
        in_position = False
        position = None
        halted = False

        df_sorted = df.sort("timestamp")
        df_sorted = df_sorted.with_row_index("idx")
        closes = df_sorted["close"].to_numpy()
        highs = df_sorted["high"].to_numpy()
        lows = df_sorted["low"].to_numpy()
        probs = df_sorted[prob_col].to_numpy() if prob_col in df_sorted.columns else np.zeros(len(df_sorted))
        atrs = df_sorted["atr"].to_numpy() if "atr" in df_sorted.columns else np.zeros(len(df_sorted))
        timestamps = df_sorted["timestamp"].to_numpy()

        for i in range(len(df_sorted)):
            current_close = closes[i]
            current_high = highs[i]
            current_low = lows[i]
            current_prob = probs[i]
            current_atr = atrs[i]
            current_time = str(timestamps[i])

            if halted:
                continue

            peak_capital = max(self.initial_capital, capital)
            current_drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
            if current_drawdown >= self.max_drawdown_pct:
                logger.warning(f"Circuit breaker triggered: drawdown {current_drawdown*100:.1f}% >= {self.max_drawdown_pct*100:.1f}%")
                halted = True
                if in_position and position is not None:
                    exit_price = current_close
                    pnl, fee, slippage = self._calculate_trade_pnl(
                        position, exit_price, position.size
                    )
                    trades.append(TradeResult(
                        entry_time=position.entry_time,
                        exit_time=current_time,
                        direction=position.direction,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        tp_price=position.tp_price,
                        sl_price=position.sl_price,
                        size=position.size,
                        pnl=pnl,
                        pnl_pct=pnl / position.size * 100 if position.size > 0 else 0,
                        result="loss",
                        fee_paid=fee,
                        slippage_paid=slippage,
                    ))
                    capital += pnl
                    in_position = False
                    position = None
                continue

            if in_position and position is not None:
                tp_hit, sl_hit = self._check_exit_conditions(position, current_high, current_low)
                trailing_hit = False

                if self.trailing_stop_enabled and not (tp_hit or sl_hit):
                    trailing_hit = self._check_trailing_stop(position, current_high, current_low)

                if tp_hit:
                    exit_price = position.tp_price
                    pnl, fee, slippage = self._calculate_trade_pnl(position, exit_price, position.size)
                    trades.append(TradeResult(
                        entry_time=position.entry_time,
                        exit_time=current_time,
                        direction=position.direction,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        tp_price=position.tp_price,
                        sl_price=position.sl_price,
                        size=position.size,
                        pnl=pnl,
                        pnl_pct=pnl / position.size * 100 if position.size > 0 else 0,
                        result="win",
                        fee_paid=fee,
                        slippage_paid=slippage,
                    ))
                    capital += pnl
                    in_position = False
                    position = None
                elif sl_hit or trailing_hit:
                    exit_price = position.sl_price if sl_hit else position.trailing_stop_price
                    pnl, fee, slippage = self._calculate_trade_pnl(position, exit_price, position.size)
                    result = "loss" if sl_hit else ("win" if pnl > 0 else "loss")
                    trades.append(TradeResult(
                        entry_time=position.entry_time,
                        exit_time=current_time,
                        direction=position.direction,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        tp_price=position.tp_price,
                        sl_price=position.sl_price,
                        size=position.size,
                        pnl=pnl,
                        pnl_pct=pnl / position.size * 100 if position.size > 0 else 0,
                        result=result,
                        fee_paid=fee,
                        slippage_paid=slippage,
                    ))
                    capital += pnl
                    in_position = False
                    position = None
                elif i - position.entry_idx >= self.forward_periods:
                    exit_price = current_close
                    pnl, fee, slippage = self._calculate_trade_pnl(position, exit_price, position.size)
                    trades.append(TradeResult(
                        entry_time=position.entry_time,
                        exit_time=current_time,
                        direction=position.direction,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        tp_price=position.tp_price,
                        sl_price=position.sl_price,
                        size=position.size,
                        pnl=pnl,
                        pnl_pct=pnl / position.size * 100 if position.size > 0 else 0,
                        result="win" if pnl > 0 else "loss",
                        fee_paid=fee,
                        slippage_paid=slippage,
                    ))
                    capital += pnl
                    in_position = False
                    position = None

            if not in_position and not halted:
                direction = None
                if current_prob >= threshold:
                    direction = "LONG"
                elif current_prob <= (1.0 - threshold):
                    direction = "SHORT"

                if direction:
                    entry_price = current_close
                    size = self._calculate_position_size(capital, entry_price, current_atr, direction)

                    if size > 0:
                        if direction == "LONG":
                            tp_price = round(entry_price * (1 + self.tp_pct), 2)
                            sl_price = round(entry_price * (1 - self.sl_pct), 2)
                        else:
                            tp_price = round(entry_price * (1 - self.tp_pct), 2)
                            sl_price = round(entry_price * (1 + self.sl_pct), 2)

                        position = Position(
                            direction=direction,
                            entry_price=entry_price,
                            entry_time=current_time,
                            entry_idx=i,
                            tp_price=tp_price,
                            sl_price=sl_price,
                            size=size,
                            atr=current_atr,
                            trailing_stop_active=False,
                            trailing_stop_price=0.0,
                        )
                        in_position = True

        closed_trades = [t for t in trades if t.result != "open"]
        wins = [t for t in closed_trades if t.result == "win"]
        losses = [t for t in closed_trades if t.result == "loss"]

        winrate = len(wins) / len(closed_trades) * 100 if closed_trades else 0.0
        total_pnl = sum(t.pnl for t in closed_trades)
        total_fees = sum(t.fee_paid for t in closed_trades)
        total_slippage = sum(t.slippage_paid for t in closed_trades)
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0
        profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')
        max_drawdown = self._calculate_max_drawdown(closed_trades, self.initial_capital)

        return {
            "total_trades": len(closed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "winrate": round(winrate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "final_capital": round(capital, 2),
            "return_pct": round((capital - self.initial_capital) / self.initial_capital * 100, 2),
            "total_fees": round(total_fees, 2),
            "total_slippage": round(total_slippage, 2),
            "trades": closed_trades,
        }

    def _calculate_position_size(self, capital: float, entry_price: float, atr: float, direction: str) -> float:
        """
        Menghitung ukuran posisi berdasarkan ATR atau fixed percentage.

        :return: Ukuran posisi dalam satuan base asset
        """
        if self.position_size_method == "atr" and atr > 0:
            risk_per_unit = atr * self.atr_multiplier
            max_risk_amount = capital * self.max_risk_per_trade
            size = max_risk_amount / risk_per_unit
            return min(size, capital * 0.95 / entry_price)
        else:
            return capital * 0.1 / entry_price

    def _calculate_trade_pnl(self, position, exit_price: float, size: float) -> tuple[float, float, float]:
        """Menghitung PnL termasuk fee dan slippage."""
        entry = position.entry_price
        direction = position.direction

        if direction == "LONG":
            gross_pnl = (exit_price - entry) * size
        else:
            gross_pnl = (entry - exit_price) * size

        fee = (entry * size + exit_price * size) * self.transaction_fee_pct
        slippage = entry * size * self.slippage_pct + exit_price * size * self.slippage_pct
        net_pnl = gross_pnl - fee - slippage

        return net_pnl, fee, slippage

    def _check_exit_conditions(self, position, current_high: float, current_low: float) -> tuple[bool, bool]:
        """Cek apakah TP atau SL hit."""
        tp_hit = (position.direction == "LONG" and current_high >= position.tp_price) or \
                 (position.direction == "SHORT" and current_low <= position.tp_price)
        sl_hit = (position.direction == "LONG" and current_low <= position.sl_price) or \
                 (position.direction == "SHORT" and current_high >= position.sl_price)
        return tp_hit, sl_hit

    def _check_trailing_stop(self, position, current_high: float, current_low: float) -> bool:
        """
        Cek trailing stop. Jika harga sudah mencapai activation threshold,
        trailing stop akan mengikuti harga.
        """
        if position.direction == "LONG":
            profit_pct = (current_high - position.entry_price) / position.entry_price
            if profit_pct >= self.trailing_stop_activation_pct:
                new_trailing = current_low * (1 - self.trailing_stop_distance_pct)
                if new_trailing > position.trailing_stop_price:
                    position.trailing_stop_price = new_trailing
                    position.sl_price = new_trailing
                    position.trailing_stop_active = True
            return position.trailing_stop_active and current_low <= position.trailing_stop_price
        else:
            profit_pct = (position.entry_price - current_low) / position.entry_price
            if profit_pct >= self.trailing_stop_activation_pct:
                new_trailing = current_high * (1 + self.trailing_stop_distance_pct)
                if new_trailing < position.trailing_stop_price or position.trailing_stop_price == 0:
                    position.trailing_stop_price = new_trailing
                    position.sl_price = new_trailing
                    position.trailing_stop_active = True
            return position.trailing_stop_active and current_high >= position.trailing_stop_price

    def _calculate_max_drawdown(self, trades: list[TradeResult], initial_capital: float) -> float:
        """Menghitung maximum drawdown dari daftar trade."""
        capital = initial_capital
        peak = initial_capital
        max_dd = 0.0

        for trade in trades:
            capital += trade.pnl
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def print_backtest_report(self, result: dict, title: str = "BACKTEST REPORT"):
        """Mencetak laporan backtest dalam format tabel."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(title)
        logger.info("=" * 80)
        logger.info(f"Total Trades      : {result['total_trades']}")
        logger.info(f"Wins              : {result['wins']}")
        logger.info(f"Losses            : {result['losses']}")
        logger.info(f"Winrate           : {result['winrate']}%")
        logger.info(f"Total PnL         : {result['total_pnl']}")
        logger.info(f"Return            : {result['return_pct']}%")
        logger.info(f"Profit Factor     : {result['profit_factor']}")
        logger.info(f"Avg Win           : {result['avg_win_pct']}%")
        logger.info(f"Avg Loss          : {result['avg_loss_pct']}%")
        logger.info(f"Max Drawdown      : {result['max_drawdown_pct']}%")
        logger.info(f"Final Capital     : {result['final_capital']}")
        logger.info(f"Total Fees        : {result['total_fees']}")
        logger.info(f"Total Slippage    : {result['total_slippage']}")
        logger.info("=" * 80)
