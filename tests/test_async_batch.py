import asyncio
from pathlib import Path
import time

import pytest

from llmrouter.adapters.base import ErrorClass, NormalizedResponse, ProviderError
from llmrouter.adapters.fake import ScriptedFakeAdapter
from llmrouter.config import RouterConfig
from llmrouter.router import Router
from llmrouter.store.sqlite import SQLiteStore


class AsyncAdapter:
    name = "async_test"

    def __init__(self):
        self.calls = 0

    def complete(self, req, model):
        self.calls += 1
        content = req.messages[-1].content
        time.sleep(0.03 if content == "slow" else 0.001)
        if content == "fail":
            raise ProviderError("deliberate failure", ErrorClass.SERVER_ERROR)
        return NormalizedResponse(text=content.upper(), input_tokens=1, output_tokens=1)

    async def acomplete(self, req, model):
        self.calls += 1
        content = req.messages[-1].content
        await asyncio.sleep(0.03 if content == "slow" else 0.001)
        if content == "fail":
            raise ProviderError("deliberate failure", ErrorClass.SERVER_ERROR)
        return NormalizedResponse(text=content.upper(), input_tokens=1, output_tokens=1)

    @staticmethod
    def classify_error(exc):
        if isinstance(exc, ProviderError):
            return exc.error_class
        return ErrorClass.SERVER_ERROR


def make_router(path: Path, adapter: AsyncAdapter) -> Router:
    config = RouterConfig.model_validate({
        "tiers": ["T0"],
        "models": [{
            "id": "async", "provider": "async_test", "model_name": "async",
            "api_key_env": "UNUSED", "tier": "T0", "enabled": True,
            "context_window": 8192, "max_output_tokens": 32,
            "input_price_per_mtok": 0, "output_price_per_mtok": 0,
            "is_free_tier": True, "data_policy": "no_train",
        }],
        "task_routes": {"answer": {
            "tier_order": ["T0"], "requires": {}, "default_temperature": 0,
            "max_attempts": 1, "budget_group": "default", "sensitivity": "public",
            "cache_ttl_seconds": None,
        }},
        "budgets": {"default": {"monthly_cap_usd": 10, "soft_warn_pct": 80, "hard_stop": True}},
    })
    return Router(config, SQLiteStore(path), adapters={"async_test": adapter})


@pytest.mark.asyncio
async def test_acall_runs_provider_work_without_blocking_event_loop(tmp_path: Path) -> None:
    adapter = AsyncAdapter()
    router = make_router(tmp_path / "async.db", adapter)
    try:
        call = asyncio.create_task(router.acall("answer", [{"role": "user", "content": "hello"}]))
        await asyncio.sleep(0)
        result = await call
        assert result.text == "HELLO"
        assert adapter.calls == 1
    finally:
        router.close()


@pytest.mark.asyncio
async def test_batch_preserves_order_reports_item_errors_and_progress(tmp_path: Path) -> None:
    adapter = AsyncAdapter()
    router = make_router(tmp_path / "batch.db", adapter)
    progress: list[tuple[int, int, int]] = []
    try:
        results = await router.batch(
            [
                [{"role": "user", "content": "slow"}],
                [{"role": "user", "content": "fast"}],
                [{"role": "user", "content": "fail"}],
            ],
            task="answer", concurrency=2,
            on_progress=lambda done, total, item: progress.append((done, total, item.index)),
        )
        assert [item.index for item in results] == [0, 1, 2]
        assert [item.result.text if item.result else None for item in results] == ["SLOW", "FAST", None]
        assert results[0].ok and results[1].ok and not results[2].ok
        assert len(progress) == 3
        assert sorted(done for done, _, _ in progress) == [1, 2, 3]
    finally:
        router.close()


@pytest.mark.asyncio
async def test_rerunning_batch_reuses_cached_successes(tmp_path: Path) -> None:
    template = make_router(tmp_path / "unused.db", AsyncAdapter())
    config = template.config
    template.close()
    fake = ScriptedFakeAdapter({
        "fake": [
            NormalizedResponse(text="one", input_tokens=1, output_tokens=1),
            NormalizedResponse(text="two", input_tokens=1, output_tokens=1),
        ]
    })
    # Use a separate configured adapter key/model ID for the cache-resume scenario.
    config.models[0].id = "fake"
    config.models[0].provider = "fake"
    router = Router(config, SQLiteStore(tmp_path / "resume.db"), adapters={"fake": fake})
    items = [
        [{"role": "user", "content": "resume one"}],
        [{"role": "user", "content": "resume two"}],
    ]
    try:
        first = await router.batch(items, task="answer", concurrency=2)
        second = await router.batch(items, task="answer", concurrency=2)
        assert all(item.ok and not item.result.cached for item in first)
        assert all(item.ok and item.result.cached for item in second)
        assert len(fake.calls) == 2
    finally:
        router.close()
