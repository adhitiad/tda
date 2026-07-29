"""
Fase 13 - Zero-Cost On-Chain & Derivatives Tracker (DefiLlama + Coinglass).
Tracks Smart Money flows, CEX transparency, and liquidation events.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger("smart_money")


class SmartMoneyTracker:
    """Zero-cost on-chain tracker using DefiLlama and Coinglass free APIs."""

    def __init__(self, coinglass_api_key: str = "") -> None:
        self.coinglass_api_key = coinglass_api_key
        self.defillama_base = "https://api.llama.fi"
        self.coinglass_base = "https://open-api.coinglass.com"

    # ------------------------------------------------------------------
    # DefiLlama: CEX Transparency
    # ------------------------------------------------------------------
    async def fetch_cex_transparency(self) -> dict[str, Any]:
        """
        Fetch CEX balance data from DefiLlama.
        Tracks stablecoin and BTC balances on major exchanges.
        Gracefully degrades if API is unavailable.
        """
        # DefiLlama stablecoins endpoint (free, no auth)
        url = "https://api.llama.fi/stablecoins"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

                # Extract total stablecoin supply as proxy for CEX buying power
                total_stablecoin = 0.0
                if isinstance(data, dict) and "data" in data:
                    for sc in data["data"][:20]:
                        try:
                            total_stablecoin += float(sc.get("totalSupply", 0.0) or 0.0)
                        except (ValueError, TypeError):
                            continue

                return {
                    "total_stablecoin_balance": total_stablecoin,
                    "total_btc_balance": 0.0,  # Not available from this endpoint
                    "exchanges": {},
                    "source": "defillama",
                }
        except Exception as e:
            logger.debug(f"[SmartMoney] DefiLlama fetch failed: {e}")
            return {
                "total_stablecoin_balance": 0.0,
                "total_btc_balance": 0.0,
                "exchanges": {},
                "source": "defillama",
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Coinglass: Open Interest & Liquidations
    # ------------------------------------------------------------------
    async def fetch_open_interest(self, symbol: str = "BTC") -> dict[str, Any]:
        """
        Fetch aggregated Open Interest from Coinglass.
        Gracefully degrades if no API key is set.
        """
        if not self.coinglass_api_key:
            logger.debug("[SmartMoney] No Coinglass API key set, skipping OI fetch")
            return {
                "open_interest_usd": 0.0,
                "oi_change_24h_pct": 0.0,
                "symbol": symbol,
                "source": "coinglass",
            }
        url = f"{self.coinglass_base}/api/futures/openInterest"
        headers = {"accept": "application/json", "CG-API-KEY": self.coinglass_api_key}
        params = {"symbol": symbol}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                oi_value = float(data.get("data", {}).get("openInterest", 0.0))
                oi_change_24h = float(data.get("data", {}).get("oiChangePercent", 0.0))
                return {
                    "open_interest_usd": oi_value,
                    "oi_change_24h_pct": oi_change_24h,
                    "symbol": symbol,
                    "source": "coinglass",
                }
        except Exception as e:
            logger.debug(f"[SmartMoney] Coinglass OI fetch failed: {e}")
            return {
                "open_interest_usd": 0.0,
                "oi_change_24h_pct": 0.0,
                "symbol": symbol,
                "source": "coinglass",
                "error": str(e),
            }

    async def fetch_liquidations(self, symbol: str = "BTC") -> dict[str, Any]:
        """
        Fetch recent liquidation data from Coinglass.
        Gracefully degrades if no API key is set.
        """
        if not self.coinglass_api_key:
            logger.debug("[SmartMoney] No Coinglass API key set, skipping liquidation fetch")
            return {
                "long_liquidation_usd": 0.0,
                "short_liquidation_usd": 0.0,
                "total_liquidation_usd": 0.0,
                "symbol": symbol,
                "source": "coinglass",
            }
        url = f"{self.coinglass_base}/api/futures/liquidation/info"
        headers = {"accept": "application/json", "CG-API-KEY": self.coinglass_api_key}
        params = {"symbol": symbol, "time_type": "h4"}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                long_liq = float(data.get("data", {}).get("longLiquidationUsd", 0.0))
                short_liq = float(data.get("data", {}).get("shortLiquidationUsd", 0.0))
                return {
                    "long_liquidation_usd": long_liq,
                    "short_liquidation_usd": short_liq,
                    "total_liquidation_usd": long_liq + short_liq,
                    "symbol": symbol,
                    "source": "coinglass",
                }
        except Exception as e:
            logger.debug(f"[SmartMoney] Coinglass liquidation fetch failed: {e}")
            return {
                "long_liquidation_usd": 0.0,
                "short_liquidation_usd": 0.0,
                "total_liquidation_usd": 0.0,
                "symbol": symbol,
                "source": "coinglass",
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Whale Alert via Telegram/RSS Scraping
    # ------------------------------------------------------------------
    async def scrape_whale_alert_rss(self) -> dict[str, Any]:
        """
        Scrape whale alert data from public RSS/Twitter feeds.
        Uses regex to extract transaction amounts from text.
        """
        try:
            import feedparser

            feeds = [
                "https://nitter.net/whale_alert/rss",
                "https://whale-alert.io/feed",
            ]
            whale_transactions: list[dict[str, Any]] = []

            for feed_url in feeds:
                try:
                    parsed = feedparser.parse(feed_url)
                    for entry in parsed.entries[:10]:
                        title = entry.get("title", "")
                        summary = entry.get("summary", "")
                        text = f"{title}. {summary}"

                        # Regex to extract: "X,XXX BTC transferred to Binance"
                        btc_match = re.search(
                            r"([\d,]+)\s*BTC\s*(?:transferred|moved|deposited)\s*to\s*(\w+)",
                            text,
                            re.IGNORECASE,
                        )
                        if btc_match:
                            amount_str = btc_match.group(1).replace(",", "")
                            exchange = btc_match.group(2)
                            whale_transactions.append({
                                "amount_btc": float(amount_str),
                                "exchange": exchange,
                                "direction": "inflow",
                                "text": title,
                            })

                        # Also check for USDT transfers
                        usdt_match = re.search(
                            r"([\d,]+)\s*USDT\s*(?:transferred|moved|deposited)\s*to\s*(\w+)",
                            text,
                            re.IGNORECASE,
                        )
                        if usdt_match:
                            amount_str = usdt_match.group(1).replace(",", "")
                            exchange = usdt_match.group(2)
                            whale_transactions.append({
                                "amount_usdt": float(amount_str),
                                "exchange": exchange,
                                "direction": "inflow",
                                "text": title,
                            })
                except Exception:
                    continue

            total_btc_inflow = sum(
                t["amount_btc"] for t in whale_transactions if t.get("amount_btc", 0) > 1000
            )
            return {
                "whale_transactions": whale_transactions[:5],
                "total_btc_inflow_large": total_btc_inflow,
                "source": "whale_alert_rss",
            }
        except Exception as e:
            logger.error(f"[SmartMoney] Whale alert scraping failed: {e}")
            return {
                "whale_transactions": [],
                "total_btc_inflow_large": 0.0,
                "source": "whale_alert_rss",
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Composite Snapshot
    # ------------------------------------------------------------------
    async def get_smart_money_snapshot(self, symbol: str = "BTC") -> dict[str, Any]:
        """
        Compile a full smart money snapshot from all sources.
        """
        cex_data = await self.fetch_cex_transparency()
        oi_data = await self.fetch_open_interest(symbol)
        liq_data = await self.fetch_liquidations(symbol)
        whale_data = await self.scrape_whale_alert_rss()

        # Classify signals
        btc_inflow_threshold = 2000.0  # BTC
        stablecoin_inflow_threshold = 500_000_000.0  # USD
        long_liq_threshold = 50_000_000.0  # USD
        oi_drop_threshold = -10.0  # percent

        btc_balance = cex_data.get("total_btc_balance", 0.0)
        stablecoin_balance = cex_data.get("total_stablecoin_balance", 0.0)
        oi_change = oi_data.get("oi_change_24h_pct", 0.0)
        long_liq = liq_data.get("long_liquidation_usd", 0.0)
        whale_btc_inflow = whale_data.get("total_btc_inflow_large", 0.0)

        # Signal classification
        cex_signal = "NEUTRAL"
        if btc_balance > btc_inflow_threshold or whale_btc_inflow > btc_inflow_threshold:
            cex_signal = "BEARISH"  # BTC inflow = potential dump
        elif stablecoin_balance > stablecoin_inflow_threshold:
            cex_signal = "BULLISH"  # Stablecoin inflow = buying power

        oi_signal = "NEUTRAL"
        if oi_change < oi_drop_threshold:
            oi_signal = "BEARISH"  # OI dropping = unwinding

        liq_signal = "NEUTRAL"
        if long_liq > long_liq_threshold:
            liq_signal = "MASSIVE_LONG_LIQUIDATION"  # Bottom fishing opportunity

        return {
            "cex_transparency": {
                "total_btc_balance": btc_balance,
                "total_stablecoin_balance": stablecoin_balance,
                "signal": cex_signal,
            },
            "open_interest": {
                "oi_usd": oi_data.get("open_interest_usd", 0.0),
                "oi_change_24h_pct": oi_change,
                "signal": oi_signal,
            },
            "liquidations": {
                "long_liquidation_usd": long_liq,
                "short_liquidation_usd": liq_data.get("short_liquidation_usd", 0.0),
                "signal": liq_signal,
            },
            "whale_alerts": {
                "total_btc_inflow_large": whale_btc_inflow,
                "transactions": whale_data.get("whale_transactions", []),
            },
            "composite_signal": self._compute_composite(cex_signal, oi_signal, liq_signal),
        }

    @staticmethod
    def _compute_composite(cex: str, oi: str, liq: str) -> str:
        bearish_count = sum(1 for s in (cex, oi) if s == "BEARISH")
        if liq == "MASSIVE_LONG_LIQUIDATION":
            return "BOTTOM_FISHING_OPPORTUNITY"
        if bearish_count >= 2:
            return "STRONGLY_BEARISH"
        if bearish_count == 1:
            return "BEARISH"
        if cex == "BULLISH":
            return "BULLISH"
        return "NEUTRAL"

    # ------------------------------------------------------------------
    # VETO Logic
    # ------------------------------------------------------------------
    def evaluate_veto(
        self,
        signal_direction: str,
        snapshot: dict[str, Any],
        is_oversold: bool = False,
    ) -> tuple[bool, str, str]:
        """
        Evaluate smart money veto.

        Returns:
            (should_veto: bool, reason: str, mode: str)
            mode can be: "DEFENSIVE", "SNIPER", or "NONE"
        """
        cex_signal = snapshot.get("cex_transparency", {}).get("signal", "NEUTRAL")
        oi_signal = snapshot.get("open_interest", {}).get("signal", "NEUTRAL")
        liq_signal = snapshot.get("liquidations", {}).get("signal", "NEUTRAL")
        composite = snapshot.get("composite_signal", "NEUTRAL")

        btc_balance = snapshot.get("cex_transparency", {}).get("total_btc_balance", 0.0)
        long_liq = snapshot.get("liquidations", {}).get("long_liquidation_usd", 0.0)
        whale_inflow = snapshot.get("whale_alerts", {}).get("total_btc_inflow_large", 0.0)

        # VETO 1: Supply Shock - Cancel LONG if BTC inflow massive or OI dropping
        if signal_direction == "LONG":
            if cex_signal == "BEARISH" or oi_signal == "BEARISH" or whale_inflow > 2000:
                reason = (
                    f"🚨 SMART MONEY ALERT: Sinyal LONG Dibatalkan! "
                    f"DefiLlama mendeteksi Inflow {btc_balance:.0f} BTC ke Bursa "
                    f"& Coinglass melaporkan ${long_liq / 1e6:.1f}M Long Liquidation. "
                    f"Sistem masuk mode Defensif."
                )
                return True, reason, "DEFENSIVE"

        # VETO 2: Bottom Fishing - Activate SNIPER mode on massive long liquidation + oversold
        if liq_signal == "MASSIVE_LONG_LIQUIDATION" and is_oversold:
            reason = (
                f"🎯 SNIPER MODE ACTIVATED: Massive Long Liquidation ${long_liq / 1e6:.1f}M "
                f"deteksi + harga oversold. Bottom fishing opportunity!"
            )
            return True, reason, "SNIPER"

        # VETO 3: Cancel SHORT if stablecoin inflow massive (buying power)
        if signal_direction == "SHORT" and cex_signal == "BULLISH":
            reason = (
                f"🚨 SMART MONEY ALERT: Sinyal SHORT Dibatalkan! "
                f"DefiLlama mendeteksi Inflow Stablecoin masif ke bursa (buying power). "
                f"Sistem masuk mode Defensif."
            )
            return True, reason, "DEFENSIVE"

        return False, "", "NONE"