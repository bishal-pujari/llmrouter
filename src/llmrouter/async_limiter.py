"""Async quota and cooldown coordination for runtimes without threads."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from llmrouter.config import ModelConfig
from llmrouter.limiter import WINDOW_SECONDS, window_start
from llmrouter.store.async_protocol import AsyncStore


@dataclass
class AsyncLimitReservation:
    limiter: "AsyncRateLimiter"
    model: ModelConfig
    starts: dict[str, int]
    estimated_input_tokens: int
    estimated_output_tokens: int
    semaphore: asyncio.Semaphore
    finished: bool = False

    async def settle(self, input_tokens: int, output_tokens: int, cost_usd: float = 0.0) -> None:
        await self._finish(input_tokens, output_tokens, cost_usd)

    async def release(self) -> None:
        await self._finish(0, 0, 0.0)

    async def _finish(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        if self.finished:
            return
        self.finished = True
        try:
            await self.limiter.store.reconcile_model_usage(
                self.model.id, self.starts, self.estimated_input_tokens,
                self.estimated_output_tokens, max(0, input_tokens), max(0, output_tokens), cost_usd,
            )
        finally:
            self.semaphore.release()


class AsyncRateLimiter:
    """Use D1 for durable quotas and an isolate-local semaphore for concurrency."""

    def __init__(self, store: AsyncStore, safety_fraction: float = 0.9) -> None:
        if not 0 < safety_fraction <= 1:
            raise ValueError("safety_fraction must be greater than 0 and at most 1")
        self.store = store
        self.safety_fraction = safety_fraction
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def _semaphore(self, model: ModelConfig) -> asyncio.Semaphore:
        value = self._semaphores.get(model.id)
        if value is None:
            value = asyncio.Semaphore(model.max_concurrency)
            self._semaphores[model.id] = value
        return value

    async def reserve(self, model: ModelConfig, estimated_input_tokens: int,
                      max_output_tokens: int, *, now: int | None = None) -> AsyncLimitReservation | None:
        timestamp = int(time.time()) if now is None else now
        semaphore = self._semaphore(model)
        if not semaphore.locked():
            await semaphore.acquire()
        else:
            return None
        starts = {name: window_start(name, timestamp) for name in WINDOW_SECONDS}
        try:
            has_headroom = await self.store.reserve_model_usage(
                model.id, starts,
                {"second": self._limit(model.rps), "minute": self._limit(model.rpm), "day": self._limit(model.rpd)},
                {"minute": self._limit(model.tpm), "day": self._limit(model.tpd)},
                estimated_input_tokens, max_output_tokens,
            )
            if not has_headroom:
                semaphore.release()
                return None
            return AsyncLimitReservation(self, model, starts, estimated_input_tokens, max_output_tokens, semaphore)
        except Exception:
            semaphore.release()
            raise

    def _limit(self, value: int | None) -> int | None:
        return None if value is None else max(1, int(value * self.safety_fraction))

    async def is_cooling_down(self, model_id: str, *, now: int | None = None) -> bool:
        timestamp = int(time.time()) if now is None else now
        return await self.store.read_cooldown(model_id, timestamp) is not None

    async def set_cooldown(self, model_id: str, cooldown_until: int, reason: str, *, consecutive_failures: int) -> None:
        await self.store.write_cooldown(model_id, cooldown_until, reason, consecutive_failures)

    async def clear_cooldown(self, model_id: str) -> None:
        await self.store.delete_cooldown(model_id)

    async def headroom_score(self, model: ModelConfig, *, now: int | None = None) -> float:
        timestamp = int(time.time()) if now is None else now
        scores: list[float] = []
        for window, limit in (("second", model.rps), ("minute", model.rpm), ("day", model.rpd)):
            if limit is None:
                continue
            row = await self.store.read_usage_window(model.id, window, window_start(window, timestamp))
            scores.append(max(0.0, (limit - (row[0] if row else 0)) / limit))
        return min(scores, default=1.0)

    async def status(self, models: list[ModelConfig]) -> list[dict[str, object]]:
        now = int(time.time())
        status: list[dict[str, object]] = []
        for model in models:
            usage = {
                window: await self.store.read_usage_window(model.id, window, window_start(window, now))
                for window in ("second", "minute", "day")
            }
            cooldown = await self.store.read_cooldown(model.id, now)
            status.append({
                "id": model.id, "provider": model.provider, "tier": model.tier,
                "enabled": model.enabled, "health": "cooldown" if cooldown else "unknown",
                "cooldown_until": cooldown[0] if cooldown else None, "usage": usage,
            })
        return status
