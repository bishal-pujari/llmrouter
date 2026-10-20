"""Retry and fallback policy, kept separate from providers and persistence."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from llmrouter.adapters.base import (
    ErrorClass,
    NormalizedRequest,
    NormalizedResponse,
    ProviderAdapter,
    ProviderError,
)
from llmrouter.config import ModelConfig
from llmrouter.errors import LLMRouterError


@dataclass
class Attempt:
    model_id: str
    attempt_no: int
    status: str
    error: str | None = None
    response: NormalizedResponse | None = None


@dataclass
class ExecutionResult:
    response: NormalizedResponse
    model_id: str
    attempts: list[Attempt] = field(default_factory=list)


class AllAttemptsFailed(LLMRouterError):
    def __init__(self, message: str, attempts: list[Attempt]) -> None:
        super().__init__(message)
        self.attempts = attempts


def execute(
    request: NormalizedRequest,
    candidates: Sequence[ModelConfig],
    adapters: dict[str, ProviderAdapter],
    *,
    max_attempts: int,
    backoff_base_s: float = 1.0,
    backoff_cap_s: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
) -> ExecutionResult:
    """Try candidates in order; retry server faults once on the same model.

    Injected clock functions make backoff deterministic in tests and keep the
    policy independent from any specific provider SDK.
    """
    history: list[Attempt] = []
    used_attempts = 0
    for model in candidates:
        if used_attempts >= max_attempts:
            break
        adapter = adapters.get(model.provider)
        if adapter is None:
            raise LLMRouterError(f"No adapter registered for provider {model.provider!r}")
        model_attempts = 0
        while model_attempts == 0 or model_attempts == 1:
            if used_attempts >= max_attempts:
                break
            used_attempts += 1
            model_attempts += 1
            try:
                response = adapter.complete(request, model)
                history.append(Attempt(model.id, used_attempts, "ok", response=response))
                return ExecutionResult(response=response, model_id=model.id, attempts=history)
            except Exception as exc:
                error_class = adapter.classify_error(exc)
                message = _safe_error(exc)
                history.append(Attempt(model.id, used_attempts, error_class.value, error=message))
                if error_class == ErrorClass.SERVER_ERROR and model_attempts == 1 and used_attempts < max_attempts:
                    delay_cap = min(backoff_cap_s, backoff_base_s * (2 ** (used_attempts - 1)))
                    sleep(max(0.0, random_value()) * delay_cap)
                    # Loop once more on this model; another failure falls back.
                    continue
                # Rate limits, timeouts, client/auth errors, and the second 5xx
                # move to the next configured candidate; cooldowns are recorded later.
                break
    raise AllAttemptsFailed("All configured model attempts failed", history)


def _safe_error(exc: Exception) -> str:
    """Bound error history and redact common credential-shaped strings."""
    message = str(exc)
    for marker in ("sk-", "Bearer ", "key="):
        position = message.lower().find(marker.lower())
        if position >= 0:
            message = message[:position] + "[redacted]"
    return message[:500]
