"""
Tests for task queue.
"""

import asyncio

import pytest

from crypto_trading_framework.ml.task_queue import BoundedTaskPool, TaskQueueConfig


class TestTaskQueueConfig:
    def test_defaults(self):
        cfg = TaskQueueConfig()
        assert cfg.enabled is True
        assert cfg.max_concurrency == 4
        assert cfg.timeout == 120.0

    def test_custom(self):
        cfg = TaskQueueConfig(enabled=False, max_concurrency=2, timeout=60.0)
        assert cfg.enabled is False
        assert cfg.max_concurrency == 2
        assert cfg.timeout == 60.0


class TestBoundedTaskPool:
    @pytest.mark.asyncio
    async def test_map_returns_results(self):
        pool = BoundedTaskPool(TaskQueueConfig(max_concurrency=2))

        async def task(x):
            return x * 2

        results = await pool.map(task, [1, 2, 3])
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_map_empty_items(self):
        pool = BoundedTaskPool(TaskQueueConfig())

        async def task(x):
            return x

        results = await pool.map(task, [])
        assert results == []

    @pytest.mark.asyncio
    async def test_map_respects_concurrency(self):
        pool = BoundedTaskPool(TaskQueueConfig(max_concurrency=1))
        concurrent = []

        async def task(x):
            concurrent.append(x)
            if len(concurrent) > 1:
                raise AssertionError("Concurrency exceeded")
            await asyncio.sleep(0.01)
            concurrent.remove(x)
            return x

        results = await pool.map(task, [1, 2, 3])
        assert results == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_map_handles_exceptions(self):
        pool = BoundedTaskPool(TaskQueueConfig(max_concurrency=2))

        async def task(x):
            if x == 2:
                raise ValueError("fail")
            return x

        results = await pool.map(task, [1, 2, 3])
        assert results[0] == 1
        assert results[1] is None
        assert results[2] == 3

    @pytest.mark.asyncio
    async def test_map_disabled_returns_empty(self):
        pool = BoundedTaskPool(TaskQueueConfig(enabled=False))

        async def task(x):
            return x

        results = await pool.map(task, [1, 2, 3])
        assert results == []

    @pytest.mark.asyncio
    async def test_map_timeout(self):
        pool = BoundedTaskPool(TaskQueueConfig(max_concurrency=2, timeout=0.05))

        async def slow_task(x):
            await asyncio.sleep(0.2)
            return x

        results = await pool.map(slow_task, [1, 2])
        assert results[0] is None
        assert results[1] is None
