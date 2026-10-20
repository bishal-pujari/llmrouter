"""Atomic fixed-window quota reservations and model cooldown tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llmrouter.config import ModelConfig

if TYPE_CHECKING:
    from llmrouter.store.protocol import Store


WINDOW_SECONDS = {"second": 1, "minute": 60, "day": 86_400}
_semaphore_guard = threading.Lock()
_semaphores: dict[tuple[str, str], threading.BoundedSemaphore] = {}


def window_start(window_type: str, timestamp: int) -> int:
    """Return the UTC epoch second beginning the aligned fixed window."""
    try:
        width = WINDOW_SECONDS[window_type]
    except KeyError as exc:
        raise ValueError(f"Unknown usage window {window_type!r}") from exc
    return timestamp - timestamp % width


def _semaphore_for(store: "Store", model: ModelConfig) -> threading.BoundedSemaphore:
    # This process-wide semaphore coordinates Router instances sharing the DB.
    key = (store.db_path, model.id)
    with _semaphore_guard:
        semaphore = _semaphores.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(model.max_concurrency)
            _semaphores[key] = semaphore
        return semaphore


@dataclass
class LimitReservation:
    """Capacity held for one provider attempt; always settle or release it."""

    limiter: "RateLimiter"
    model: ModelConfig
    starts: dict[str, int]
    estimated_input_tokens: int
    estimated_output_tokens: int
    semaphore: threading.BoundedSemaphore
    _finished: bool = False
    _finish_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def settle(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
    ) -> None:
        """Replace estimated token usage with provider-reported usage."""
        self._finish(input_tokens, output_tokens, cost_usd)

    def release(self) -> None:
        """Release reserved tokens after a failure with no billable usage."""
        self._finish(0, 0, 0.0)

    def _finish(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        # A per-instance lock prevents double settlement by competing cleanup paths.
        with self._finish_lock:
            if self._finished:
                return
            input_tokens = max(0, input_tokens)
            output_tokens = max(0, output_tokens)
            try:
                self.limiter.store.reconcile_model_usage(
                    self.model.id, self.starts, self.estimated_input_tokens,
                    self.estimated_output_tokens, input_tokens, output_tokens, cost_usd,
                )
            finally:
                self._finished = True
                self.semaphore.release()

    def __enter__(self) -> "LimitReservation":
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if not self._finished:
            self.release()


class RateLimiter:
    """Tracks configured quota windows in SQLite and active calls in-process."""

    def __init__(self, store: "Store", safety_fraction: float = 0.9) -> None:
        if not 0 < safety_fraction <= 1:
            raise ValueError("safety_fraction must be greater than 0 and at most 1")
        self.store = store
        self.safety_fraction = safety_fraction

    def reserve(
        self,
        model: ModelConfig,
        estimated_input_tokens: int,
        max_output_tokens: int,
        *,
        now: int | None = None,
    ) -> LimitReservation | None:
        """Atomically reserve request/token headroom; return None when exhausted."""
        if estimated_input_tokens < 0 or max_output_tokens < 0:
            raise ValueError("token estimates cannot be negative")
        timestamp = int(time.time()) if now is None else now
        starts = {name: window_start(name, timestamp) for name in WINDOW_SECONDS}
        semaphore = _semaphore_for(self.store, model)
        if not semaphore.acquire(blocking=False):
            return None

        try:
            request_limits = {
                "second": self._effective_limit(model.rps) if model.rps is not None else None,
                "minute": self._effective_limit(model.rpm) if model.rpm is not None else None,
                "day": self._effective_limit(model.rpd) if model.rpd is not None else None,
            }
            token_limits = {
                "minute": self._effective_limit(model.tpm) if model.tpm is not None else None,
                "day": self._effective_limit(model.tpd) if model.tpd is not None else None,
            }
            has_headroom = self.store.reserve_model_usage(
                model.id, starts, request_limits, token_limits,
                estimated_input_tokens, max_output_tokens,
            )
            if not has_headroom:
                semaphore.release()
                return None
            return LimitReservation(
                limiter=self,
                model=model,
                starts=starts,
                estimated_input_tokens=estimated_input_tokens,
                estimated_output_tokens=max_output_tokens,
                semaphore=semaphore,
            )
        except Exception:
            semaphore.release()
            raise

    def _effective_limit(self, configured_limit: int) -> int:
        # A positive provider quota must permit at least one request/token unit;
        # otherwise flooring 90% of a small integer limit would disable it.
        return max(1, int(configured_limit * self.safety_fraction))

    def is_cooling_down(self, model_id: str, *, now: int | None = None) -> bool:
        timestamp = int(time.time()) if now is None else now
        return self.store.read_cooldown(model_id, timestamp) is not None

    def set_cooldown(
        self,
        model_id: str,
        cooldown_until: int,
        reason: str,
        *,
        consecutive_failures: int = 1,
    ) -> None:
        self.store.write_cooldown(model_id, cooldown_until, reason, consecutive_failures)

    def clear_cooldown(self, model_id: str) -> None:
        self.store.delete_cooldown(model_id)

    def usage(self, model: ModelConfig, *, now: int | None = None) -> dict[str, dict[str, int]]:
        timestamp = int(time.time()) if now is None else now
        result: dict[str, dict[str, int]] = {}
        for window in WINDOW_SECONDS:
            row = self.store.read_usage_window(model.id, window, window_start(window, timestamp))
            result[window] = (
                {"requests": 0, "input_tokens": 0, "output_tokens": 0}
                if row is None else dict(zip(("requests", "input_tokens", "output_tokens"), row))
            )
        return result

    def status(self, models: list[ModelConfig], *, now: int | None = None) -> list[dict[str, object]]:
        timestamp = int(time.time()) if now is None else now
        statuses = []
        for model in models:
            cooldown = self.store.read_cooldown(model.id, timestamp)
            statuses.append({
                "model_id": model.id,
                "usage": self.usage(model, now=timestamp),
                "limits": {"rps": model.rps, "rpm": model.rpm, "rpd": model.rpd,
                           "tpm": model.tpm, "tpd": model.tpd,
                           "max_concurrency": model.max_concurrency},
                "cooldown": None if cooldown is None else {
                    "until": cooldown[0], "reason": cooldown[1], "consecutive_failures": cooldown[2]
                },
            })
        return statuses

    def headroom_score(self, model: ModelConfig, *, now: int | None = None) -> float:
        """Return the least remaining quota fraction across configured windows."""
        usage = self.usage(model, now=now)
        fractions = []
        for window, limit in (("second", model.rps), ("minute", model.rpm), ("day", model.rpd)):
            if limit is not None:
                fractions.append(max(0.0, 1 - usage[window]["requests"] / limit))
        for window, limit in (("minute", model.tpm), ("day", model.tpd)):
            if limit is not None:
                used = usage[window]["input_tokens"] + usage[window]["output_tokens"]
                fractions.append(max(0.0, 1 - used / limit))
        return min(fractions, default=1.0)
