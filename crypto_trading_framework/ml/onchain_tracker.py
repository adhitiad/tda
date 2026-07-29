"""
Fase 13 - On-Chain Whale Tracker & Exchange Flow Integration.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger("onchain")


class OnChainTracker:
    """Tracks whale transactions and exchange netflow."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self.cryptoquant_base = "https://api.cryptoquant.com/v1"
        self.whale_alert_base = "https://api.whale-alert.io/v1"

    async def fetch_exchange_netflow(self, symbol: str = "btc") -> dict[str, Any]:
        """
        Fetch exchange netflow (inflow - outflow) for the last 24 hours.
        Returns value in native asset units (BTC/ETH).
        Gracefully degrades if no API key is set.
        """
        if not self.api_key:
            logger.debug("[OnChain] No CryptoQuant API key set, skipping netflow fetch")
            return {"exchange_netflow_24h": "N/A (no API key)", "netflow_value": 0.0}
        url = f"{self.cryptoquant_base}/exchange-flows/netflow"
        params = {
            "symbol": symbol,
            "window": "24h",
            "api_key": self.api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                netflow = data.get("data", {}).get("netflow", 0.0)
                return {
                    "exchange_netflow_24h": f"{'+' if netflow > 0 else ''}{netflow:.0f} {symbol.upper()}",
                    "netflow_value": netflow,
                }
        except Exception as e:
            logger.error(f"[OnChain] Failed to fetch netflow for {symbol}: {e}")
            return {"exchange_netflow_24h": "N/A", "netflow_value": 0.0}

    async def fetch_whale_transactions(self, symbol: str = "btc", min_value_usd: float = 50_000_000) -> dict[str, Any]:
        """
        Fetch large whale transactions (> $50M) involving exchanges.
        Returns transaction count and total volume.
        Gracefully degrades if no API key is set.
        """
        if not self.api_key:
            logger.debug("[OnChain] No Whale Alert API key set, skipping whale tx fetch")
            return {"whale_transactions_count": 0, "whale_total_volume_usd": 0.0}
        url = f"{self.whale_alert_base}/transactions"
        params = {
            "symbol": symbol,
            "min_value": min_value_usd,
            "api_key": self.api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                transactions = data.get("data", {}).get("transactions", [])
                total_volume = sum(t.get("amount_usd", 0.0) for t in transactions)
                return {
                    "whale_transactions_count": len(transactions),
                    "whale_total_volume_usd": total_volume,
                }
        except Exception as e:
            logger.error(f"[OnChain] Failed to fetch whale transactions for {symbol}: {e}")
            return {"whale_transactions_count": 0, "whale_total_volume_usd": 0.0}

    async def get_onchain_snapshot(self, symbol: str = "btc") -> dict[str, Any]:
        """
        Compile on-chain snapshot containing netflow and whale activity.
        """
        netflow_data = await self.fetch_exchange_netflow(symbol)
        whale_data = await self.fetch_whale_transactions(symbol)

        snapshot = {
            **netflow_data,
            **whale_data,
            "symbol": symbol.upper(),
        }

        # Classify whale activity
        inflow_threshold = 2000.0 if symbol.lower() == "btc" else 20000.0
        netflow_value = netflow_data.get("netflow_value", 0.0)

        if netflow_value > inflow_threshold:
            snapshot["whale_activity"] = "High Inflow (Bearish)"
        elif netflow_value < -inflow_threshold:
            snapshot["whale_activity"] = "High Outflow (Bullish)"
        else:
            snapshot["whale_activity"] = "Normal"

        snapshot["veto_status"] = "NONE"
        return snapshot

    def evaluate_veto(self, signal_direction: str, onchain_data: dict[str, Any]) -> tuple[bool, str]:
        """
        Evaluate whether on-chain data should veto the current signal.

        Returns:
            (should_veto: bool, reason: str)
        """
        netflow_value = onchain_data.get("netflow_value", 0.0)
        whale_activity = onchain_data.get("whale_activity", "Normal")
        symbol = onchain_data.get("symbol", "BTC")

        # WHALE DUMP WARNING: Exchange Inflow extreme
        if signal_direction == "LONG" and netflow_value > 2000.0:
            reason = f"🚨 WHALE ALERT: Terdeteksi Inflow eksstrem {netflow_value:.0f} {symbol} ke bursa. Sinyal LONG dibatalkan atau direduksi drastis."
            return True, reason

        # SUPPLY SHOCK (BULLISH): Exchange Outflow extreme
        if signal_direction == "SHORT" and netflow_value < -2000.0:
            reason = f"🚨 SUPPLY SHOCK: Terdeteksi Outflow eksstrem {abs(netflow_value):.0f} {symbol} dari bursa ke cold wallet. Sinyal SHORT dibatalkan."
            return True, reason

        # Additional whale transaction check
        if whale_activity == "High Inflow (Bearish)" and signal_direction == "LONG":
            reason = f"🚨 WHALE ALERT: Aktivitas whale inflow tinggi. Sinyal LONG dibatalkan."
            return True, reason

        if whale_activity == "High Outflow (Bullish)" and signal_direction == "SHORT":
            reason = f"🚨 SUPPLY SHOCK: Aktivitas whale outflow tinggi. Sinyal SHORT dibatalkan."
            return True, reason

        return False, ""