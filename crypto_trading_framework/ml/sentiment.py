"""
Fase 8 - NLP Sentiment Analysis & Macro Emotion (Fear & Greed).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
import feedparser
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


logger = logging.getLogger("sentiment")


class SentimentEngine:
    """Aggregates Fear & Greed index and crypto news sentiment."""

    def __init__(self) -> None:
        self.analyzer = SentimentIntensityAnalyzer()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=2, max=60, jitter=2),
        reraise=False,
    )
    async def fetch_fear_greed(self) -> dict[str, Any]:
        url = "https://api.alternative.me/fng/"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    data = await resp.json()
                    current = data.get("data", [{}])[0]
                    score = int(current.get("value", 50))
                    label = current.get("value_classification", "Neutral")
                    return {
                        "score": score,
                        "label": label,
                    }
        except Exception as e:
            logger.error(f"[Sentiment] Failed to fetch Fear & Greed: {e}")
            return {"score": 50, "label": "Neutral"}

    def _classify_compound(self, compound: float) -> tuple[str, float]:
        if compound >= 0.35:
            return "Bullish", compound
        if compound <= -0.35:
            return "Bearish", compound
        return "Neutral", compound

    async def fetch_news_sentiment(self, rss_feeds: list[str] | None = None) -> dict[str, Any]:
        feeds = rss_feeds or [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/feed",
            "https://cryptopanic.com/feed/rss",
        ]
        headlines: list[str] = []
        try:
            loop = asyncio.get_event_loop()
            for feed_url in feeds:
                try:
                    parsed = await loop.run_in_executor(None, feedparser.parse, feed_url)
                    for entry in parsed.entries[:20]:
                        title = entry.get("title", "")
                        summary = entry.get("summary", "")
                        headlines.append(f"{title}. {summary}".strip())
                except Exception as e:
                    logger.debug(f"[Sentiment] RSS parse failed {feed_url}: {e}")
        except Exception as e:
            logger.error(f"[Sentiment] Failed fetching RSS: {e}")

        if not headlines:
            return {
                "compound": 0.0,
                "label": "Neutral",
                "headline_count": 0,
            }

        scores = [self.analyzer.polarity_scores(text)["compound"] for text in headlines]
        avg_compound = float(sum(scores) / len(scores))
        label, _ = self._classify_compound(avg_compound)

        return {
            "compound": round(avg_compound, 4),
            "label": label,
            "headline_count": len(headlines),
        }

    def adjust_probability(
        self,
        technical_probability: float,
        fear_greed_score: int,
        news_compound: float,
    ) -> tuple[float, float]:
        """
        Adjust technical probability using fundamental scores.

        Returns (adjusted_probability, adjustment_pct).
        """
        base = technical_probability
        fg_adj = 0.0
        news_adj = 0.0

        if fear_greed_score >= 75:
            fg_adj = -0.10
        elif fear_greed_score <= 25:
            fg_adj = +0.10

        if news_compound >= 0.5:
            news_adj = +0.05
        elif news_compound <= -0.5:
            news_adj = -0.05

        total_adj = fg_adj + news_adj
        adjusted = max(0.0, min(1.0, base + total_adj))
        adjustment_pct = (adjusted - base) * 100.0

        return adjusted, adjustment_pct