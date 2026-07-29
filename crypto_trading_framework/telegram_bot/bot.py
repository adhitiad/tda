from __future__ import annotations

import asyncio
from typing import Any

from telegram.ext import (
    Application,
)

from crypto_trading_framework.core.logging import get_logger
from crypto_trading_framework.telegram_bot.config import TelegramBotConfig
from crypto_trading_framework.telegram_bot.formatter import SignalFormatter
from crypto_trading_framework.telegram_bot.handler import TelegramHandler
from crypto_trading_framework.telegram_bot.rate_limiter import RateLimiter
from crypto_trading_framework.telegram_bot.router import SignalRouter

logger = get_logger("telegram_bot")


class TelegramBot:
    def __init__(self, config: TelegramBotConfig) -> None:
        self.config = config
        self.application: Application | None = None
        self.router = SignalRouter(config.groups)
        self.formatter = SignalFormatter(
            {
                "disclaimer_text": config.signal.disclaimer_text,
                "timezone": config.signal.timezone,
            },
        )
        self.rate_limiter = RateLimiter(
            {
                "max_signals_per_hour": 5,
                "max_messages_per_minute": 10,
                "cooldown_minutes_per_symbol": 15,
                "burst_window_minutes": 5,
                "burst_max_count": 3,
            },
        )
        self.handler = TelegramHandler(
            config,
            self.router,
            self.formatter,
            self.rate_limiter,
        )
        self._running = False

    async def start(self) -> None:
        if not self.config.enabled or not self.config.bot_token:
            logger.warning("Telegram bot tidak diaktifkan atau token kosong.")
            return
        try:
            self.application = Application.builder().token(self.config.bot_token).build()
            self._register_handlers()
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling(
                poll_interval=self.config.polling_interval,
                allowed_updates=["message", "callback_query"],
            )
            logger.info(f"Telegram bot dimulai untuk {len(self.config.groups)} grup.")
            self._running = True
        except (OSError, RuntimeError) as e:
            logger.error(f"Gagal memulai Telegram bot: {e}")
            self._running = False

    async def stop(self) -> None:
        if self.application is None:
            return
        try:
            await self.application.updater.stop()
        except (asyncio.CancelledError, OSError):
            pass
        try:
            await self.application.stop()
        except (asyncio.CancelledError, OSError):
            pass
        try:
            await self.application.shutdown()
        except (asyncio.CancelledError, OSError):
            pass
        self._running = False
        logger.info("Telegram bot dihentikan.")

    def _register_handlers(self) -> None:
        if not self.application:
            return
        for cmd_handler in self.handler.get_handlers():
            self.application.add_handler(cmd_handler)
        self.application.add_handler(self.handler.get_message_handler())
        logger.info(f"{len(self.handler.get_handlers())} command handler dan 1 message handler terdaftar.")

    async def send_signal(
        self,
        symbol: str,
        asset_type: str,
        signal_data: dict[str, Any],
    ) -> int:
        groups = self.router.route_signal(symbol, asset_type)
        if not groups:
            logger.debug(f"Tidak ada grup yang menerima sinyal untuk {symbol} ({asset_type})")
            return 0
        sent_count = 0
        for group in groups:
            group_id = group.group_id
            if not self.rate_limiter.can_send_signal(group_id, symbol, self._group_config(group)):
                logger.debug(f"Rate limit terpicu untuk {symbol} di grup {group_id}")
                continue
            if not self.rate_limiter.can_send_message(group_id, self._group_config(group)):
                logger.debug(f"Message rate limit terpicu untuk grup {group_id}")
                continue
            message = self._format_signal(asset_type, signal_data, group)
            if not message:
                continue
            try:
                if self.application and self.application.bot:
                    await self.application.bot.send_message(
                        chat_id=group_id,
                        text=message,
                        parse_mode="HTML",
                    )
                    self.rate_limiter.record_signal(group_id, symbol)
                    self.rate_limiter.record_message(group_id)
                    sent_count += 1
                    logger.info(f"Sinyal {symbol} dikirim ke grup {group.name}")
            except (OSError, RuntimeError) as e:
                logger.error(f"Gagal mengirim sinyal ke grup {group_id}: {e}")
        return sent_count

    def _format_signal(
        self,
        asset_type: str,
        signal_data: dict[str, Any],
        group: Any,
    ) -> str:
        if asset_type == "crypto":
            win_rate_val = 0.0
            winrate_str = signal_data.get("winrate", "")
            if winrate_str and "%" in winrate_str:
                import re as _re
                match = _re.search(r"(\d+)\s*-\s*(\d+)%", winrate_str)
                if match:
                    low = float(match.group(1))
                    high = float(match.group(2))
                    win_rate_val = (low + high) / 2.0
            return self.formatter.format_entry_signal(
                symbol=signal_data.get("symbol", signal_data.get("simbol", "")),
                direction=signal_data.get("direction", "LONG"),
                entry=signal_data.get("entry", signal_data.get("entry_zone", [0.0])[0] if signal_data.get("entry_zone") else 0.0),
                targets=signal_data.get("take_profit", [0.0]) if isinstance(signal_data.get("take_profit"), list) else [signal_data.get("take_profit", 0.0)],
                stop_loss=signal_data.get("stop_loss", signal_data.get("stop_loss_atr", 0.0)),
                timeframe=signal_data.get("timeframe", ""),
                indicators=signal_data.get("indicators"),
                confidence=signal_data.get("probability", "MEDIUM"),
                risk_reward=0.0,
                tier=group.tier.value,
                win_rate=win_rate_val,
                current_price=signal_data.get("current_price", 0.0),
            )
        return ""

    def _group_config(self, group: Any) -> dict[str, Any]:
        return {
            "max_signals_per_hour": group.max_signals_per_hour,
            "max_messages_per_minute": group.max_messages_per_minute,
            "cooldown_minutes_per_symbol": group.cooldown_minutes_per_symbol,
            "burst_window_minutes": group.burst_window_minutes,
            "burst_max_count": group.burst_max_count,
        }

    @property
    def is_running(self) -> bool:
        return self._running