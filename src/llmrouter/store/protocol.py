"""Storage boundary: core routing code should depend on this protocol only."""

from typing import Protocol
from dataclasses import dataclass

from llmrouter.config import RouterConfig


@dataclass
class CallLogEntry:
    id: str
    ts: int
    request_id: str
    task: str
    caller: str | None
    model_id: str
    attempt_no: int
    status: str
    http_status: int | None
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float
    shadow_cost_usd: float
    error_message: str | None
    prompt_hash: str


class Store(Protocol):
    db_path: str

    def close(self) -> None: ...

    def upsert_config(self, config: RouterConfig) -> None: ...

    def reserve_model_usage(
        self, model_id: str, starts: dict[str, int], request_limits: dict[str, int | None],
        token_limits: dict[str, int | None], input_tokens: int, output_tokens: int,
    ) -> bool: ...

    def reconcile_model_usage(
        self, model_id: str, starts: dict[str, int], estimated_input: int, estimated_output: int,
        actual_input: int, actual_output: int, cost_usd: float,
    ) -> None: ...

    def read_usage_window(self, model_id: str, window: str, start: int) -> tuple[int, int, int] | None: ...

    def read_cooldown(self, model_id: str, now: int) -> tuple[int, str, int] | None: ...

    def write_cooldown(
        self, model_id: str, cooldown_until: int, reason: str, consecutive_failures: int
    ) -> None: ...

    def delete_cooldown(self, model_id: str) -> None: ...

    def log_call(self, entry: CallLogEntry) -> None: ...

    def reserve_budget(
        self, reservation_id: str, request_id: str, budget_group: str,
        worst_case_usd: float, expires_at: int,
    ) -> tuple[float, float, float]: ...

    def release_budget(self, reservation_id: str, *, expired: bool = False) -> None: ...

    def settle_budget_and_log(self, reservation_id: str, entry: CallLogEntry) -> None: ...

    def cache_get(self, key: str, *, now: int | None = None) -> dict[str, object] | None: ...

    def cache_put(
        self, key: str, task: str, model_family: str,
        response: dict[str, object], ttl_seconds: int | None,
    ) -> None: ...

    def cache_clear(self, task: str | None = None) -> int: ...

    def cache_stats(self, *, now: int | None = None) -> dict[str, object]: ...

    def budget_report(self, month: str, start: int, end: int) -> dict[str, object]: ...
