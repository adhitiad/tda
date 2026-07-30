"""Asynchronous crypto news ingestion from Binance News API and CoinDesk RSS."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import aiohttp
import feedparser

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("news_ingestion")

BINANCE_NEWS_URL = "https://api.binance.com/sapi/v1/publisher/news"
COINDESK_RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"

BINANCE_LIMIT = 5
COINDESK_LIMIT = 5
MAX_TOTAL = 10
FETCH_TIMEOUT_SECONDS = 10


def _title_hash(title: str) -> str:
    return hashlib.sha256(title.strip().lower().encode("utf-8")).hexdigest()


async def _fetch_binance_news(session: aiohttp.ClientSession) -> list[dict[str, str]]:
    try:
        async with session.get(
            BINANCE_NEWS_URL,
            params={"limit": BINANCE_LIMIT},
            timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"[News] Binance API returned status {resp.status}")
                return []
            data: dict[str, Any] = await resp.json()
            articles = data.get("data", []) if isinstance(data, dict) else []
            results: list[dict[str, str]] = []
            for article in articles[:BINANCE_LIMIT]:
                title = article.get("title", "")
                url = article.get("url", "")
                if title and url:
                    results.append({"title": title, "url": url})
            logger.info(f"[News] Fetched {len(results)} Binance articles")
            return results
    except asyncio.TimeoutError:
        logger.error("[News] Binance News API request timed out")
        return []
    except Exception as exc:
        logger.error(f"[News] Binance fetch error: {exc}")
        return []


async def _fetch_coindesk_rss() -> list[dict[str, str]]:
    try:
        loop = asyncio.get_event_loop()
        parsed = await loop.run_in_executor(None, feedparser.parse, COINDESK_RSS_URL)
        results: list[dict[str, str]] = []
        for entry in parsed.entries[:COINDESK_LIMIT]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            if title and link:
                results.append({"title": title, "url": link})
        logger.info(f"[News] Fetched {len(results)} CoinDesk headlines")
        return results
    except Exception as exc:
        logger.error(f"[News] CoinDesk RSS parse error: {exc}")
        return []


async def fetch_latest_crypto_news() -> list[dict[str, str]]:
    """Fetch and deduplicate crypto headlines from Binance News API and CoinDesk RSS.

    Returns a list of up to MAX_TOTAL items, each with 'title' and 'url' keys.
    Deduplication is performed on the SHA-256 hash of the lowercase, stripped title.
    """
    async with aiohttp.ClientSession() as session:
        binance_task = asyncio.create_task(_fetch_binance_news(session))
        coindesk_task = asyncio.create_task(_fetch_coindesk_rss())
        binance_news, coindesk_news = await asyncio.gather(
            binance_task, coindesk_task, return_exceptions=True
        )

    all_items: list[dict[str, str]] = []
    seen_hashes: set[str] = set()

    for batch in (binance_news, coindesk_news):
        if isinstance(batch, Exception):
            logger.error(f"[News] Batch fetch failed: {batch}")
            continue
        for item in batch:
            h = _title_hash(item["title"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_items.append(item)

    result = all_items[:MAX_TOTAL]
    logger.info(f"[News] Returning {len(result)} deduplicated headlines")
    return result