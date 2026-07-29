from __future__ import annotations

from crypto_trading_framework.core.logging import get_logger
from crypto_trading_framework.telegram_bot.config import (
    AssetType,
    GroupTier,
    TelegramGroupConfig,
)

logger = get_logger("telegram_router")


class SignalRouter:
    def __init__(self, groups: list[TelegramGroupConfig]) -> None:
        self.groups = groups
        self._tier_index: dict[GroupTier, list[TelegramGroupConfig]] = {}
        for group in groups:
            self._tier_index.setdefault(group.tier, []).append(group)

    def route_signal(self, symbol: str, asset_type: str) -> list[TelegramGroupConfig]:
        matched: list[TelegramGroupConfig] = []
        for group in self.groups:
            if self._group_accepts_asset(group, symbol, asset_type):
                matched.append(group)
        return matched

    def _group_accepts_asset(
        self,
        group: TelegramGroupConfig,
        symbol: str,
        asset_type: str,
    ) -> bool:
        if group.asset_type == AssetType.CRYPTO.value:
            return asset_type == AssetType.CRYPTO.value and self._symbol_in_group(symbol, group)
        if group.asset_type == AssetType.CRYPTO_FOREX.value:
            return asset_type in (AssetType.CRYPTO.value, AssetType.FOREX.value) and self._symbol_in_group(symbol, group)
        if group.asset_type == AssetType.CRYPTO_IDX.value:
            return asset_type in (AssetType.CRYPTO.value, AssetType.IDX.value) and self._symbol_in_group(symbol, group)
        if group.asset_type == AssetType.CRYPTO_FOREX_IDX.value:
            return asset_type in (AssetType.CRYPTO.value, AssetType.FOREX.value, AssetType.IDX.value) and self._symbol_in_group(symbol, group)
        return False

    def _symbol_in_group(self, symbol: str, group: TelegramGroupConfig) -> bool:
        if not group.allowed_symbols:
            return True
        return any(symbol.startswith(s) or symbol == s for s in group.allowed_symbols)

    def get_groups_by_tier(self, tier: GroupTier) -> list[TelegramGroupConfig]:
        return self._tier_index.get(tier, [])

    def get_all_groups(self) -> list[TelegramGroupConfig]:
        return list(self.groups)

    def find_group(self, group_id: str) -> TelegramGroupConfig | None:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        return None