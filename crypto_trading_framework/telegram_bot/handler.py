from __future__ import annotations

import re
from typing import Any

from telegram import Update
from telegram.ext import CommandHandler, ContextTypes, MessageHandler, filters

from crypto_trading_framework.core.logging import get_logger
from crypto_trading_framework.telegram_bot.config import TelegramBotConfig
from crypto_trading_framework.telegram_bot.formatter import SignalFormatter
from crypto_trading_framework.telegram_bot.rate_limiter import RateLimiter
from crypto_trading_framework.telegram_bot.router import SignalRouter

logger = get_logger("telegram_handler")


class TelegramHandler:
    def __init__(
        self,
        bot_config: TelegramBotConfig,
        router: SignalRouter,
        formatter: SignalFormatter,
        rate_limiter: RateLimiter,
    ) -> None:
        self.bot_config = bot_config
        self.router = router
        self.formatter = formatter
        self.rate_limiter = rate_limiter
        self._command_pattern = re.compile(r"^/(\w+)(?:\s+(.*))?$")

    def get_handlers(self) -> list[CommandHandler]:
        return [
            CommandHandler("start", self.handle_start),
            CommandHandler("help", self.handle_help),
            CommandHandler("status", self.handle_status),
            CommandHandler("signal", self.handle_signal),
            CommandHandler("watchlist", self.handle_watchlist),
            CommandHandler("history", self.handle_history),
            CommandHandler("config", self.handle_config),
            CommandHandler("digest", self.handle_digest),
        ]

    def get_message_handler(self) -> MessageHandler:
        return MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text)

    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return
        group_id = str(update.effective_chat.id)
        group = self.router.find_group(group_id)
        if group is None:
            await update.message.reply_text("❌ Grup ini tidak terdaftar di konfigurasi bot.")
            return
        await update.message.reply_text(
            self.formatter.format_welcome(group.name, group.tier.value),
        )
        self.rate_limiter.record_message(group_id)

    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        await update.message.reply_text(self.formatter.format_command_help())

    async def handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return
        group_id = str(update.effective_chat.id)
        group = self.router.find_group(group_id)
        if group is None:
            await update.message.reply_text("❌ Grup ini tidak terdaftar di konfigurasi bot.")
            return
        remaining = self.rate_limiter.get_remaining_slots(group_id, self._group_config(group))
        lines = [
            "📊 Status Bot",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📋 Grup: {group.name}",
            f"🏷 Tier: {group.tier.value.upper()}",
            f"📦 Asset Type: {group.asset_type}",
            f"🔢 Sinyal tersisa/jam: {remaining}",
            f"📡 Bot aktif: {self.bot_config.enabled}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        await update.message.reply_text("\n".join(lines))
        self.rate_limiter.record_message(group_id)

    async def handle_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return
        group_id = str(update.effective_chat.id)
        group = self.router.find_group(group_id)
        if group is None:
            await update.message.reply_text("❌ Grup ini tidak terdaftar di konfigurasi bot.")
            return
        if not context.args:
            await update.message.reply_text(
                "❌ Gunakan format: /signal {symbol}\nContoh: /signal BTC/USDT",
            )
            return
        symbol = context.args[0].upper().strip()
        if not self._symbol_allowed_for_group(symbol, group):
            await update.message.reply_text(
                f"🚫 Grup '{group.name}' (tier: {group.tier.value}) tidak mendukung sinyal untuk {symbol}.",
            )
            return
        if not self.rate_limiter.can_send_message(group_id, self._group_config(group)):
            await update.message.reply_text(
                self.formatter.format_rate_limit(30),
            )
            return
        await update.message.reply_text(
            f"🔍 Menganalisis {symbol}... Sinyal akan dikirim jika tersedia.",
        )
        self.rate_limiter.record_message(group_id)

    async def handle_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return
        group_id = str(update.effective_chat.id)
        group = self.router.find_group(group_id)
        if group is None:
            await update.message.reply_text("❌ Grup ini tidak terdaftar di konfigurasi bot.")
            return
        symbols = group.allowed_symbols if group.allowed_symbols else ["Semua crypto"]
        lines = [
            f"📋 Watchlist — {group.name}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        for s in symbols:
            lines.append(f"  • {s}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        await update.message.reply_text("\n".join(lines))
        self.rate_limiter.record_message(group_id)

    async def handle_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        group_id = str(update.effective_chat.id)
        group = self.router.find_group(group_id)
        if group is None:
            await update.message.reply_text("❌ Grup ini tidak terdaftar di konfigurasi bot.")
            return
        if group.tier.value not in ("pro", "enterprise"):
            await update.message.reply_text(
                "🚫 Riwayat sinyal hanya tersedia untuk tier Pro dan Enterprise.",
            )
            return
        await update.message.reply_text("📜 Fitur riwayat sinyal sedang dalam pengembangan.")
        self.rate_limiter.record_message(group_id)

    async def handle_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return
        group_id = str(update.effective_chat.id)
        group = self.router.find_group(group_id)
        if group is None:
            await update.message.reply_text("❌ Grup ini tidak terdaftar di konfigurasi bot.")
            return
        if group.tier.value != "enterprise":
            await update.message.reply_text(
                "🚫 Konfigurasi grup hanya dapat diakses oleh tier Enterprise.",
            )
            return
        lines = [
            f"⚙ Konfigurasi — {group.name}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Tier: {group.tier.value}",
            f"Asset Type: {group.asset_type}",
            f"Max Sinyal/Jam: {group.max_signals_per_hour}",
            f"Cooldown/Symbol: {group.cooldown_minutes_per_symbol} menit",
            f"Burst Limit: {group.burst_max_count} dalam {group.burst_window_minutes} menit",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        await update.message.reply_text("\n".join(lines))
        self.rate_limiter.record_message(group_id)

    async def handle_digest(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        group_id = str(update.effective_chat.id)
        group = self.router.find_group(group_id)
        if group is None:
            await update.message.reply_text("❌ Grup ini tidak terdaftar di konfigurasi bot.")
            return
        if group.tier.value not in ("pro", "enterprise"):
            await update.message.reply_text(
                "🚫 Ringkasan harian hanya tersedia untuk tier Pro dan Enterprise.",
            )
            return
        await update.message.reply_text("📰 Fitur ringkasan harian sedang dalam pengembangan.")
        self.rate_limiter.record_message(group_id)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return
        group_id = str(update.effective_chat.id)
        group = self.router.find_group(group_id)
        if group is None:
            return
        text = update.message.text.strip()
        match = self._command_pattern.match(text)
        if match:
            command = match.group(1).lower()
            args = match.group(2) or ""
            await self._route_command(update, context, command, args)
        else:
            await update.message.reply_text(
                "ℹ Ketik /help untuk melihat daftar command yang tersedia.",
            )
        self.rate_limiter.record_message(group_id)

    async def _route_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        command: str,
        args: str,
    ) -> None:
        command_map: dict[str, Any] = {
            "start": self.handle_start,
            "help": self.handle_help,
            "status": self.handle_status,
            "signal": self.handle_signal,
            "watchlist": self.handle_watchlist,
            "history": self.handle_history,
            "config": self.handle_config,
            "digest": self.handle_digest,
        }
        handler = command_map.get(command)
        if handler:
            fake_args = args.split() if args else []
            context.args = fake_args
            await handler(update, context)
        else:
            await update.message.reply_text(
                "❓ Command tidak dikenal. Ketik /help untuk daftar command.",
            )

    def _symbol_allowed_for_group(self, symbol: str, group: Any) -> bool:
        if not group.allowed_symbols:
            return True
        return any(symbol.startswith(s) or symbol == s for s in group.allowed_symbols)

    def _group_config(self, group: Any) -> dict[str, Any]:
        return {
            "max_signals_per_hour": group.max_signals_per_hour,
            "max_messages_per_minute": group.max_messages_per_minute,
            "cooldown_minutes_per_symbol": group.cooldown_minutes_per_symbol,
            "burst_window_minutes": group.burst_window_minutes,
            "burst_max_count": group.burst_max_count,
        }