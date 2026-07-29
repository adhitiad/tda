from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("telegram_rate_limiter")


class RateLimiter:
    def __init__(self, default_config: dict[str, Any] | None = None) -> None:
        self.default_config = default_config or {}
        self._signals: dict[str, list[float]] = defaultdict(list)
        self._messages: dict[str, list[float]] = defaultdict(list)
        self._symbol_cooldowns: dict[str, float] = {}
        self._burst_counts: dict[str, list[float]] = defaultdict(list)

    def can_send_signal(self, group_id: str, symbol: str, config: dict[str, Any] | None = None) -> bool:
        cfg = config or self.default_config
        max_per_hour = cfg.get("max_signals_per_hour", 5)
        cooldown_min = cfg.get("cooldown_minutes_per_symbol", 15)
        burst_window = cfg.get("burst_window_minutes", 5)
        burst_max = cfg.get("burst_max_count", 3)

        now = time.time()

        if not self._check_cooldown(group_id, symbol, now, cooldown_min):
            return False

        if not self._check_burst(group_id, now, burst_window, burst_max):
            return False

        self._cleanup_old_entries(group_id, now, 3600)
        hour_count = len(self._signals.get(group_id, []))
        return not hour_count >= max_per_hour

    def can_send_message(self, group_id: str, config: dict[str, Any] | None = None) -> bool:
        cfg = config or self.default_config
        max_per_minute = cfg.get("max_messages_per_minute", 10)

        now = time.time()
        self._cleanup_old_messages(group_id, now, 60)
        minute_count = len(self._messages.get(group_id, []))
        return minute_count < max_per_minute

    def record_signal(self, group_id: str, symbol: str) -> None:
        now = time.time()
        self._signals.setdefault(group_id, []).append(now)
        self._symbol_cooldowns[f"{group_id}:{symbol}"] = now
        self._burst_counts.setdefault(group_id, []).append(now)

    def record_message(self, group_id: str) -> None:
        self._messages.setdefault(group_id, []).append(time.time())

    def _check_cooldown(
        self,
        group_id: str,
        symbol: str,
        now: float,
        cooldown_min: int,
    ) -> bool:
        key = f"{group_id}:{symbol}"
        last_sent = self._symbol_cooldowns.get(key, 0.0)
        elapsed = (now - last_sent) / 60.0
        if elapsed < cooldown_min:
            remaining = int(cooldown_min - elapsed)
            logger.debug(f"Cooldown aktif untuk {symbol} di {group_id}: {remaining} menit tersisa")
            return False
        return True

    def _check_burst(
        self,
        group_id: str,
        now: float,
        burst_window: int,
        burst_max: int,
    ) -> bool:
        window_start = now - (burst_window * 60)
        recent = [t for t in self._burst_counts.get(group_id, []) if t > window_start]
        self._burst_counts[group_id] = recent
        if len(recent) >= burst_max:
            logger.warning(f"Burst limit terpicu untuk grup {group_id}: {len(recent)} sinyal dalam {burst_window} menit")
            return False
        return True

    def _cleanup_old_entries(self, group_id: str, now: float, window_seconds: int) -> None:
        cutoff = now - window_seconds
        self._signals[group_id] = [t for t in self._signals.get(group_id, []) if t > cutoff]

    def _cleanup_old_messages(self, group_id: str, now: float, window_seconds: int) -> None:
        cutoff = now - window_seconds
        self._messages[group_id] = [t for t in self._messages.get(group_id, []) if t > cutoff]

    def get_remaining_slots(self, group_id: str, config: dict[str, Any] | None = None) -> int:
        cfg = config or self.default_config
        max_per_hour = cfg.get("max_signals_per_hour", 5)
        now = time.time()
        self._cleanup_old_entries(group_id, now, 3600)
        hour_count = len(self._signals.get(group_id, []))
        return max(0, max_per_hour - hour_count)

    def reset(self, group_id: str | None = None) -> None:
        if group_id:
            self._signals.pop(group_id, None)
            self._messages.pop(group_id, None)
            self._burst_counts.pop(group_id, None)
        else:
            self._signals.clear()
            self._messages.clear()
            self._burst_counts.clear()
            self._symbol_cooldowns.clear()