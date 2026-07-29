from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class GroupTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class AssetType(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    IDX = "idx"
    CRYPTO_FOREX = "crypto_forex"
    CRYPTO_IDX = "crypto_idx"
    CRYPTO_FOREX_IDX = "crypto_forex_idx"


class TelegramGroupConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    group_id: str
    name: str
    tier: GroupTier
    asset_type: AssetType
    allowed_symbols: list[str] = Field(default_factory=list)
    max_signals_per_hour: int = 5
    max_messages_per_minute: int = 10
    cooldown_minutes_per_symbol: int = 15
    burst_window_minutes: int = 5
    burst_max_count: int = 3


class TelegramSignalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_disclaimer: bool = True
    disclaimer_text: str = "⚠ Ini adalah analisis teknikal, bukan nasihat investasi. Gunakan dengan risiko Anda sendiri."
    timezone: str = "Asia/Jakarta"
    timestamp_format: str = "%Y-%m-%d %H:%M:%S WIB"


class TelegramBotConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = Field(default_factory=lambda: os.getenv("TELEGRAM_BOT_ENABLED", "false").lower() == "true")
    bot_token: str = Field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    groups: list[TelegramGroupConfig] = Field(default_factory=list)
    signal: TelegramSignalConfig = Field(default_factory=TelegramSignalConfig)
    polling_interval: float = 1.0
    max_retries: int = 3
    retry_delay_seconds: int = 5