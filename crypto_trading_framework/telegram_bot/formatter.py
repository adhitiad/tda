from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("telegram_formatter")


class SignalFormatter:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.disclaimer = self.config.get(
            "disclaimer_text",
            "⚠ Ini adalah analisis teknikal, bukan nasihat investasi. Gunakan dengan risiko Anda sendiri.",
        )
        self.timezone = self.config.get("timezone", "Asia/Jakarta")

    def format_entry_signal(
        self,
        symbol: str,
        direction: str,
        entry: float,
        targets: list[float],
        stop_loss: float,
        timeframe: str,
        indicators: dict[str, Any] | None = None,
        confidence: str = "MEDIUM",
        risk_reward: float = 0.0,
        tier: str = "free",
        win_rate: float = 0.0,
    ) -> str:
        lines: list[str] = []
        lines.append(f"🔔 SINYAL {tier.upper()} — {symbol}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"📌 Pasangan: {symbol}")
        lines.append(f"📊 Arah: {direction}")
        lines.append(f"💰 Entry: {entry}")
        if len(targets) >= 1:
            lines.append(f"🎯 Target 1: {targets[0]}")
        if len(targets) >= 2:
            lines.append(f"🎯 Target 2: {targets[1]}")
        if len(targets) >= 3:
            lines.append(f"🎯 Target 3: {targets[2]}")
        lines.append(f"🛑 Stop Loss: {stop_loss}")
        lines.append(f"⏱ Timeframe: {timeframe}")
        if indicators:
            ind_parts = ", ".join(f"{k}: {v}" for k, v in indicators.items())
            lines.append(f"📈 Indikator: {ind_parts}")
        lines.append(f"🎯 Confidence: {confidence}")
        if win_rate > 0:
            lines.append(f"🎗️ Win Rate : {win_rate:.1f}%")
        if risk_reward > 0:
            lines.append(f"⚠ Risk/Reward: {risk_reward:.2f}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(self.disclaimer)
        return "\n".join(lines)

    def format_exit_signal(
        self,
        symbol: str,
        reason: str,
        exit_price: float,
        pnl_pct: float | None = None,
        timeframe: str = "",
    ) -> str:
        lines: list[str] = []
        lines.append(f"✅ EXIT {symbol}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"📌 Pasangan: {symbol}")
        lines.append(f"📊 Alasan: {reason}")
        lines.append(f"💰 Exit Price: {exit_price}")
        if pnl_pct is not None:
            emoji = "📈" if pnl_pct >= 0 else "📉"
            lines.append(f"{emoji} P&L: {pnl_pct:+.2f}%")
        if timeframe:
            lines.append(f"⏱ Timeframe: {timeframe}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    def format_price_update(
        self,
        symbol: str,
        price: float,
        change_24h: float,
        volume: float,
        high: float,
        low: float,
    ) -> str:
        change_emoji = "📈" if change_24h >= 0 else "📉"
        lines: list[str] = []
        lines.append(f"📊 UPDATE {symbol}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"💰 Harga: {price}")
        lines.append(f"{change_emoji} 24h Change: {change_24h:+.2f}%")
        lines.append(f"📊 Volume: {volume}")
        lines.append(f"🔺 High: {high} | 🔻 Low: {low}")
        lines.append(f"⏱ Waktu: {self._now()}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    def format_risk_warning(
        self,
        symbol: str,
        risk_description: str,
        volatility: str,
        drawdown: float,
        recommendation: str,
        win_rate: float = 0.0,
    ) -> str:
        lines: list[str] = []
        lines.append(f"⚠ PERINGATAN RISK — {symbol}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"📌 Pasangan: {symbol}")
        lines.append(f"⚠ Risiko: {risk_description}")
        lines.append(f"📊 Volatilitas: {volatility}")
        lines.append(f"📉 Drawdown: {drawdown:.2f}%")
        lines.append(f"💡 Rekomendasi: {recommendation}")
        if win_rate > 0:
            lines.append(f"🎗️ Win Rate : {win_rate:.1f}%")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    def format_error(self, symbol: str, message: str) -> str:
        return f"❌ Error untuk {symbol}: {message}"

    def format_rate_limit(self, seconds: int) -> str:
        return f"⏳ Rate limit aktif. Silakan tunggu {seconds} detik sebelum mengirim command berikutnya."

    def format_welcome(self, group_name: str, tier: str) -> str:
        lines: list[str] = []
        lines.append(f"👋 Selamat datang di {group_name}!")
        lines.append(f"📋 Tier: {tier.upper()}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("Command yang tersedia:")
        lines.append("/signal {symbol} — Minta sinyal manual")
        lines.append("/watchlist — Lihat asset yang dipantau")
        lines.append("/status — Status bot")
        lines.append("/help — Daftar command")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(self.disclaimer)
        return "\n".join(lines)

    def format_forex_signal(
        self,
        symbol: str,
        direction: str,
        entry: float,
        targets: list[float],
        stop_loss: float,
        timeframe: str,
        analysis: str = "",
        sentiment: str = "",
        confidence: str = "MEDIUM",
        risk_reward: float = 0.0,
        win_rate: float = 0.0,
    ) -> str:
        lines: list[str] = []
        lines.append(f"🔔 SINYAL FOREX — {symbol}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"📌 Pasangan: {symbol}")
        lines.append(f"📊 Arah: {direction}")
        lines.append(f"💰 Entry: {entry}")
        if len(targets) >= 1:
            lines.append(f"🎯 Target 1: {targets[0]}")
        if len(targets) >= 2:
            lines.append(f"🎯 Target 2: {targets[1]}")
        lines.append(f"🛑 Stop Loss: {stop_loss}")
        lines.append(f"⏱ Timeframe: {timeframe}")
        if analysis:
            lines.append(f"📈 Analisis: {analysis}")
        if sentiment:
            lines.append(f"📊 Sentimen: {sentiment}")
        lines.append(f"🎯 Confidence: {confidence}")
        if win_rate > 0:
            lines.append(f"🎗️ Win Rate : {win_rate:.1f}%")
        if risk_reward > 0:
            lines.append(f"⚠ Risk/Reward: {risk_reward:.2f}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("ℹ Pastikan mempertimbangkan sesi trading yang aktif.")
        lines.append(self.disclaimer)
        return "\n".join(lines)

    def format_idx_signal(
        self,
        index_name: str,
        ticker: str,
        direction: str,
        current_level: float,
        target_level: float,
        support: float,
        resistance: float,
        timeframe: str,
        indicators: dict[str, Any] | None = None,
        confidence: str = "MEDIUM",
        risk_reward: float = 0.0,
        win_rate: float = 0.0,
    ) -> str:
        lines: list[str] = []
        lines.append(f"🔔 SINYAL IDX — {index_name}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"📌 Indeks: {index_name} ({ticker})")
        lines.append(f"📊 Arah: {direction}")
        lines.append(f"💰 Level: {current_level}")
        lines.append(f"🎯 Target: {target_level}")
        lines.append(f"🛑 Support: {support} | Resistance: {resistance}")
        lines.append(f"⏱ Timeframe: {timeframe}")
        if indicators:
            ind_parts = ", ".join(f"{k}: {v}" for k, v in indicators.items())
            lines.append(f"📈 Indikator: {ind_parts}")
        lines.append(f"🎯 Confidence: {confidence}")
        if win_rate > 0:
            lines.append(f"🎗️ Win Rate : {win_rate:.1f}%")
        if risk_reward > 0:
            lines.append(f"⚠ Risk/Reward: {risk_reward:.2f}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("ℹ Data IDX diperbarui setiap sesi perdagangan.")
        lines.append(self.disclaimer)
        return "\n".join(lines)

    def format_command_help(self) -> str:
        lines: list[str] = []
        lines.append("📚 Daftar Command:")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("/start — Pesan selamat datang")
        lines.append("/help — Daftar command")
        lines.append("/status — Status bot dan asset")
        lines.append("/signal {symbol} — Minta sinyal manual")
        lines.append("/watchlist — Asset yang dipantau")
        lines.append("/history — Riwayat sinyal (Pro+)")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(self.disclaimer)
        return "\n".join(lines)

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0700")