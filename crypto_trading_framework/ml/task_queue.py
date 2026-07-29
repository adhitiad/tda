"""
Bounded concurrency task pool for distributed ingestion.

Uses asyncio.Semaphore to limit concurrent symbol/timeframe fetches.
No external broker required; replaces sequential symbol loops with
parallel workers while respecting exchange rate limits.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("task_queue")


@dataclass
class TaskQueueConfig:
    """Configuration for task queue."""

    model_config: dict | None = None
    enabled: bool = True
    max_concurrency: int = 4
    timeout: float = 120.0


class BoundedTaskPool:
    """Asyncio task pool with bounded concurrency."""

    def __init__(self, config: TaskQueueConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrency)

    async def map(self, func: Callable[..., Coroutine[Any, Any, Any]], items: list[Any]) -> list[Any]:
        """
        Run func(item) for each item with bounded concurrency.

        :param func: Async callable accepting one item from items.
        :param items: Items to process.
        :return: Results in input order; exceptions become None.
        """
        if not self.config.enabled or not items:
            return []

        async def _run(item):
            async with self.semaphore:
                try:
                    return await asyncio.wait_for(func(item), timeout=self.config.timeout)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"[TaskQueue] Task failed for {item}: {e}")
                    return None

        tasks = [_run(item) for item in items]
        return list(await asyncio.gather(*tasks))
