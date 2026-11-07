"""Asynchronous persistence boundary used by the Cloudflare Worker path."""

from __future__ import annotations

from typing import Protocol

from llmrouter.config import RouterConfig
from llmrouter.store.protocol import CallLogEntry


class AsyncStore(Protocol):
    """The subset of storage operations needed by :class:`AsyncRouter`."""

    db_path: str

    async def close(self) -> None: ...

    async def upsert_config(self, config: RouterConfig) -> None: ...

    async def reserve_model_usage(
        self, model_id: str, starts: dict[str, int], request_limits: dict[str, int | None],
        token_limits: dict[str, int | None], input_tokens: int, output_tokens: int,
    ) -> bool: ...

    async def reconcile_model_usage(
        self, model_id: str, starts: dict[str, int], estimated_input: int, estimated_output: int,
        actual_input: int, actual_output: int, cost_usd: float,
    ) -> None: ...

    async def read_usage_window(self, model_id: str, window: str, start: int) -> tuple[int, int, int] | None: ...

    async def read_cooldown(self, model_id: str, now: int) -> tuple[int, str, int] | None: ...

    async def write_cooldown(self, model_id: str, cooldown_until: int, reason: str, consecutive_failures: int) -> None: ...

    async def delete_cooldown(self, model_id: str) -> None: ...

    async def log_call(self, entry: CallLogEntry) -> None: ...

    async def reserve_budget(
        self, reservation_id: str, request_id: str, budget_group: str,
        worst_case_usd: float, expires_at: int,
    ) -> tuple[float, float, float]: ...

    async def release_budget(self, reservation_id: str, *, expired: bool = False) -> None: ...

    async def settle_budget_and_log(self, reservation_id: str, entry: CallLogEntry) -> None: ...

    async def cache_get(self, key: str, *, now: int | None = None) -> dict[str, object] | None: ...

    async def cache_put(
        self, key: str, task: str, model_family: str,
        response: dict[str, object], ttl_seconds: int | None,
    ) -> None: ...

    async def cache_clear(self, task: str | None = None) -> int: ...

    async def cache_stats(self, *, now: int | None = None) -> dict[str, object]: ...

    async def budget_report(self, month: str, start: int, end: int) -> dict[str, object]: ...
