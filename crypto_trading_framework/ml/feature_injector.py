"""Redis state management and Polars LazyFrame sentiment feature injection."""

from __future__ import annotations

from typing import Any

import polars as pl

from crypto_trading_framework.core.logging import get_logger
from crypto_trading_framework.db.redis_cache import get_redis_cache

logger = get_logger("feature_injector")

SENTIMENT_KEY = "market:sentiment_score"
SENTIMENT_HISTORY_KEY = "market:sentiment_history"
SENTIMENT_HISTORY_MAX = 100
SENTIMENT_TTL = 600


async def update_redis_sentiment(score: float) -> None:
    """Store the latest sentiment score in Redis with a 10-minute TTL.

    Also appends the score to a rolling history list for lag feature computation.
    """
    try:
        cache = get_redis_cache()
        cache.set(SENTIMENT_KEY, score, ttl=SENTIMENT_TTL)
        cache._client.lpush(SENTIMENT_HISTORY_KEY, str(score))
        cache._client.ltrim(SENTIMENT_HISTORY_KEY, 0, SENTIMENT_HISTORY_MAX - 1)
        logger.info(f"[FeatureInjector] Sentiment score {score:.4f} stored in Redis")
    except Exception as exc:
        logger.error(f"[FeatureInjector] Failed to update Redis sentiment: {exc}")


def _get_sentiment_history(cache: Any, n: int = 3) -> list[float]:
    try:
        raw = cache._client.lrange(SENTIMENT_HISTORY_KEY, 0, n - 1)
        scores: list[float] = []
        for item in raw:
            try:
                scores.append(float(item))
            except (ValueError, TypeError):
                continue
        return scores
    except Exception as exc:
        logger.warning(f"[FeatureInjector] Failed to read sentiment history: {exc}")
        return []


def inject_sentiment_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Inject sentiment features into a Polars LazyFrame.

    Adds three columns:
      - sentiment_current: latest sentiment score from Redis
      - sentiment_lag_1:  sentiment score from 1 period ago
      - sentiment_lag_2:  sentiment score from 2 periods ago

    Uses Redis for state and history, broadcasting constant values across all rows.
    Falls back to 0.0 for any missing sentiment data.
    """
    try:
        cache = get_redis_cache()
        current_score: float = cache.get(SENTIMENT_KEY, default=0.0)
        if not isinstance(current_score, (int, float)):
            current_score = 0.0

        history = _get_sentiment_history(cache, n=3)
        lag_1 = history[1] if len(history) > 1 else 0.0
        lag_2 = history[2] if len(history) > 2 else 0.0
    except Exception as exc:
        logger.error(f"[FeatureInjector] Redis read failed, using zero fallback: {exc}")
        current_score = 0.0
        lag_1 = 0.0
        lag_2 = 0.0

    df = df.with_columns(
        pl.lit(current_score).alias("sentiment_current"),
        pl.lit(lag_1).alias("sentiment_lag_1"),
        pl.lit(lag_2).alias("sentiment_lag_2"),
    )

    logger.debug(
        f"[FeatureInjector] Injected sentiment features "
        f"(current={current_score:.4f}, lag_1={lag_1:.4f}, lag_2={lag_2:.4f})"
    )
    return df