"""Synchronous public router API for the first provider-backed milestone."""

from __future__ import annotations

import asyncio
import inspect
import math
import hashlib
import json
import os
import random
import time
import uuid
import calendar
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from llmrouter.adapters.base import (
    ErrorClass,
    Message,
    NormalizedRequest,
    NormalizedResponse,
    ProviderAdapter,
    ProviderError,
)
from llmrouter.adapters.anthropic import AnthropicAdapter
from llmrouter.adapters.cloudflare import CloudflareAdapter
from llmrouter.adapters.gemini import GeminiAdapter
from llmrouter.adapters.openai_compat import OpenAICompatAdapter
from llmrouter.call_logging import CallLogger, default_log_path
from llmrouter.config import ModelConfig, RouterConfig, TaskRoute, load_config
from llmrouter.errors import BudgetExceeded, ConfigError, NoModelAvailable, OutputValidationError
from llmrouter.executor import AllAttemptsFailed, Attempt
from llmrouter.limiter import RateLimiter
from llmrouter.selector import ModelSelector
from llmrouter.store.protocol import CallLogEntry
from llmrouter.store.sqlite import SQLiteStore
from llmrouter.validation import (
    CustomValidator,
    JSONSchema,
    json_instruction,
    schema_as_json_schema,
    validate_json_output,
)

@dataclass
class CallResult:
    text: str
    model_id: str
    cost_usd: float
    cached: bool
    attempts: list[Attempt]
    input_tokens: int
    output_tokens: int
    request_id: str
    data: Any = None
    comparison_cost_usd: float | None = None
    savings_usd: float | None = None

    @property
    def estimated_cost_usd(self) -> float:
        """Router-calculated cost using configured rates and reported token usage."""
        return self.cost_usd


@dataclass
class BatchResult:
    """Outcome for one input passed to :meth:`Router.batch`."""

    index: int
    result: CallResult | None = None
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class _AsyncExecutionAdapter:
    """Expose an adapter to the shared executor without mutating Router state."""

    def __init__(self, adapter: ProviderAdapter) -> None:
        self._adapter = adapter
        self.name = adapter.name

    def complete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse:
        return self._adapter.complete(req, model)

    def classify_error(self, exc_or_response: object) -> ErrorClass:
        return self._adapter.classify_error(exc_or_response)


class Router:
    """Route calls using validated YAML, SQLite quota state, and provider adapters."""

    def __init__(
        self,
        config: RouterConfig,
        store: SQLiteStore,
        *,
        adapters: dict[str, ProviderAdapter] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.store.upsert_config(config)
        self.call_logger = CallLogger(
            default_log_path(store.db_path),
            secrets=[os.getenv(model.api_key_env, "") for model in config.models],
        )
        self.limiter = RateLimiter(store, config.settings.limit_safety_fraction)
        self.selector = ModelSelector(config.models, self.limiter)
        self.adapters: dict[str, ProviderAdapter] = {
            "openai_compat": OpenAICompatAdapter(),
            "anthropic": AnthropicAdapter(),
            "gemini": GeminiAdapter(),
            "cloudflare": CloudflareAdapter(),
        }
        if adapters:
            self.adapters.update(adapters)
        # Keep worker ownership scoped to this Router so close() controls its lifecycle.
        self._async_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="llmrouter")
        self._disabled_models: set[str] = set()
        self._consecutive_failures: dict[str, int] = {}

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        db_path: str | Path = "data/router.db",
        *,
        adapters: dict[str, ProviderAdapter] | None = None,
    ) -> "Router":
        config = load_config(path)
        store = SQLiteStore(db_path)
        return cls(config, store, adapters=adapters)

    def call(
        self,
        task: str,
        messages: Sequence[dict[str, str] | Message],
        *,
        caller: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        sensitivity: str | None = None,
        timeout_s: float | None = None,
        force_model: str | None = None,
        use_cache: bool = True,
        cache_ttl_seconds: int | None = None,
        max_cost_usd: float | None = None,
    ) -> CallResult:
        """Make a chat completion with configured routing, limits, and fallback."""
        return self._execute(
            task, messages, caller=caller, system=system, temperature=temperature,
            max_output_tokens=max_output_tokens, sensitivity=sensitivity, timeout_s=timeout_s,
            force_model=force_model,
            use_cache=use_cache, cache_ttl_seconds=cache_ttl_seconds,
            max_cost_usd=max_cost_usd,
        )

    def call_json(
        self,
        task: str,
        messages: Sequence[dict[str, str] | Message],
        schema: JSONSchema,
        *,
        caller: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        sensitivity: str | None = None,
        timeout_s: float | None = None,
        force_model: str | None = None,
        validator: CustomValidator | None = None,
        use_cache: bool = True,
        cache_ttl_seconds: int | None = None,
        max_cost_usd: float | None = None,
    ) -> CallResult:
        """Return schema-validated structured output, retrying a failed shape once."""
        schema_json = schema_as_json_schema(schema)
        instructions = json_instruction(schema_json, system)
        return self._execute(
            task,
            messages,
            caller=caller,
            system=instructions,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            sensitivity=sensitivity,
            timeout_s=timeout_s,
            force_model=force_model,
            use_cache=use_cache,
            cache_ttl_seconds=cache_ttl_seconds,
            max_cost_usd=max_cost_usd,
            structured_schema=schema,
            custom_validator=validator,
        )

    async def acall(self, task: str, messages: Sequence[dict[str, str] | Message], **options: Any) -> CallResult:
        """Async chat completion; synchronous routing and provider I/O run off-loop."""
        adapters = {name: _AsyncExecutionAdapter(adapter) for name, adapter in self.adapters.items()}
        future = self._async_executor.submit(
            self._execute, task, messages, _adapter_overrides=adapters, **options
        )
        # Poll rather than blocking the loop; this also keeps cancellation from
        # releasing budget/concurrency holds while the worker is still executing.
        while not future.done():
            await asyncio.sleep(0.01)
        return future.result()

    async def acall_json(
        self,
        task: str,
        messages: Sequence[dict[str, str] | Message],
        schema: JSONSchema,
        *,
        caller: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        sensitivity: str | None = None,
        timeout_s: float | None = None,
        force_model: str | None = None,
        validator: CustomValidator | None = None,
        use_cache: bool = True,
        cache_ttl_seconds: int | None = None,
        max_cost_usd: float | None = None,
    ) -> CallResult:
        """Async counterpart to :meth:`call_json`."""
        schema_json = schema_as_json_schema(schema)
        instructions = json_instruction(schema_json, system)
        adapters = {name: _AsyncExecutionAdapter(adapter) for name, adapter in self.adapters.items()}
        future = self._async_executor.submit(
            self._execute, task, messages, caller=caller, system=instructions,
            temperature=temperature, max_output_tokens=max_output_tokens,
            sensitivity=sensitivity, timeout_s=timeout_s, force_model=force_model,
            use_cache=use_cache, cache_ttl_seconds=cache_ttl_seconds,
            max_cost_usd=max_cost_usd, structured_schema=schema,
            custom_validator=validator, _adapter_overrides=adapters,
        )
        while not future.done():
            await asyncio.sleep(0.01)
        return future.result()

    async def batch(
        self,
        items: Sequence[Sequence[dict[str, str] | Message] | dict[str, Any]],
        *,
        task: str,
        concurrency: int = 8,
        on_progress: Callable[[int, int, BatchResult], object] | None = None,
        **call_options: Any,
    ) -> list[BatchResult]:
        """Run independent calls concurrently and preserve input order.

        Each item is either a messages sequence or a mapping with ``messages`` and
        optional per-item call options. Cached successful items are returned from
        the normal cache path, making reruns naturally resumable.
        """
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        gate = asyncio.Semaphore(concurrency)
        results: list[BatchResult | None] = [None] * len(items)
        completed = 0
        progress_lock = asyncio.Lock()

        async def run_one(index: int, item: Sequence[dict[str, str] | Message] | dict[str, Any]) -> None:
            nonlocal completed
            async with gate:
                try:
                    if isinstance(item, dict):
                        if "messages" not in item:
                            raise ValueError("batch mapping items must include 'messages'")
                        messages = item["messages"]
                        options = {**call_options, **item.get("options", {})}
                    else:
                        messages = item
                        options = call_options
                    outcome = BatchResult(index=index, result=await self.acall(task, messages, **options))
                except Exception as exc:
                    outcome = BatchResult(index=index, error=exc)
                results[index] = outcome
                if on_progress is not None:
                    async with progress_lock:
                        completed += 1
                        try:
                            returned = on_progress(completed, len(items), outcome)
                            if inspect.isawaitable(returned):
                                await returned
                        except Exception:
                            # A reporting callback must not discard completed work.
                            pass

        await asyncio.gather(*(run_one(index, item) for index, item in enumerate(items)))
        return [item for item in results if item is not None]

    def test_model(self, model_id: str, *, task: str | None = None) -> CallResult:
        """Send a small uncached prompt directly to one enabled configured model."""
        model = next((item for item in self.config.models if item.id == model_id), None)
        if model is None:
            raise ConfigError(f"Unknown model {model_id!r}")
        if not model.enabled:
            raise ConfigError(f"Model {model_id!r} is disabled; enable it before testing")
        task_name = task or next(iter(self.config.task_routes), None)
        if task_name is None or task_name not in self.config.task_routes:
            raise ConfigError("A configured task route is required to test a model")
        original_route = self.config.task_routes[task_name]
        test_route: TaskRoute = original_route.model_copy(update={
            "tier_order": [model.tier], "requires": {}, "default_temperature": 0,
            "max_attempts": 1, "sensitivity": "public", "cache_ttl_seconds": 0,
        })
        return self._execute(
            task_name,
            [{"role": "user", "content": "Reply with the word OK."}],
            force_model=model_id,
            max_output_tokens=min(16, model.max_output_tokens),
            use_cache=False,
            route_override=test_route,
        )

    def _execute(
        self,
        task: str,
        messages: Sequence[dict[str, str] | Message],
        *,
        caller: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        sensitivity: str | None = None,
        timeout_s: float | None = None,
        force_model: str | None = None,
        use_cache: bool = True,
        cache_ttl_seconds: int | None = None,
        max_cost_usd: float | None = None,
        structured_schema: JSONSchema | None = None,
        custom_validator: CustomValidator | None = None,
        route_override: TaskRoute | None = None,
        _adapter_overrides: dict[str, ProviderAdapter] | None = None,
    ) -> CallResult:
        """Shared provider attempt loop for text and structured calls."""
        route = route_override or self.config.task_routes.get(task)
        if route is None:
            raise ConfigError(f"Unknown task route {task!r}")
        if not messages:
            raise ValueError("messages must contain at least one message")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        if timeout_s is not None and (timeout_s <= 0 or not math.isfinite(timeout_s)):
            raise ValueError("timeout_s must be greater than 0")
        if cache_ttl_seconds is not None and cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds cannot be negative")
        if max_cost_usd is not None and (
            max_cost_usd < 0 or not math.isfinite(max_cost_usd)
        ):
            raise ValueError("max_cost_usd cannot be negative")
        normalized_messages = [Message.model_validate(item) for item in messages]
        if system:
            normalized_messages.insert(0, Message(role="system", content=system))
        request_id = str(uuid.uuid4())
        request_timeout = (
            self.config.settings.default_timeout_s if timeout_s is None else timeout_s
        )
        # Bound the whole logical call even when several provider attempts are
        # needed. Each attempt still receives up to request_timeout seconds,
        # while retry backoff consumes the same finite deadline.
        logical_deadline = time.monotonic() + request_timeout * route.max_attempts
        request_temperature = route.default_temperature if temperature is None else temperature
        if request_temperature < 0 or not math.isfinite(request_temperature):
            raise ValueError("temperature must be finite and non-negative")
        required_sensitivity = sensitivity or route.sensitivity
        ttl = route.cache_ttl_seconds if cache_ttl_seconds is None else cache_ttl_seconds
        cache_enabled = (
            use_cache and self.config.settings.cache_enabled and ttl != 0
            and custom_validator is None
        )
        cache_models = self.config.models
        if force_model is not None:
            cache_models = [model for model in self.config.models if model.id == force_model]
            if not cache_models:
                raise ConfigError(f"Unknown model {force_model!r}")
        model_family = _model_family(cache_models, route.tier_order)
        cache_key = _cache_key(
            task, normalized_messages, structured_schema, request_temperature,
            max_output_tokens, model_family, required_sensitivity,
        )
        prompt_hash = _prompt_hash(normalized_messages)
        if cache_enabled:
            cached = self.store.cache_get(cache_key)
            if cached is not None:
                model_id = str(cached["model_id"])
                cached_text = str(cached["text"])
                data = (
                    validate_json_output(cached_text, structured_schema)
                    if structured_schema is not None else None
                )
                self._record_cache_hit(
                    request_id=request_id, task=task, caller=caller, model_id=model_id,
                    prompt_hash=prompt_hash,
                )
                return CallResult(
                    text=cached_text, model_id=model_id, cost_usd=0.0, cached=True,
                    attempts=[Attempt(model_id, 0, "cache_hit")],
                    input_tokens=int(cached["input_tokens"]),
                    output_tokens=int(cached["output_tokens"]),
                    request_id=request_id, data=data,
                )
        max_attempts = route.max_attempts
        attempted_models: set[str] = set()
        retried_server_models: set[str] = set()
        history: list[Attempt] = []
        retry_model: ModelConfig | None = None
        retry_messages: list[Message] | None = None
        validation_retried_models: set[str] = set()
        validation_failures_by_tier: dict[str, int] = {}
        escalated_tiers: set[str] = set()
        last_validation_error: OutputValidationError | None = None
        call_spend_usd = 0.0

        for attempt_no in range(1, max_attempts + 1):
            current_messages = retry_messages if retry_messages is not None else normalized_messages
            retry_messages = None
            estimated_input = _estimate_tokens(current_messages)
            prompt_hash = _prompt_hash(current_messages)
            selection = None
            if retry_model is not None:
                output_limit = max_output_tokens if max_output_tokens is not None else retry_model.max_output_tokens
                reservation = self.limiter.reserve(
                    retry_model, estimated_input, output_limit
                )
                if reservation is not None:
                    selection = (retry_model, reservation)
                retry_model = None

            if selection is None:
                excluded = attempted_models | self._disabled_models
                try:
                    selected = self.selector.select_and_reserve(
                        route,
                        estimated_input_tokens=estimated_input,
                        max_output_tokens=max_output_tokens,
                        sensitivity=sensitivity,
                        excluded_model_ids=excluded,
                        excluded_tiers=escalated_tiers,
                        allowed_model_ids={force_model} if force_model is not None else None,
                    )
                    selection = (selected.model, selected.reservation)
                except NoModelAvailable:
                    if history:
                        if last_validation_error is not None and history[-1].status == "invalid_output":
                            raise OutputValidationError(
                                str(last_validation_error), last_validation_error.problems, history
                            )
                        last = history[-1]
                        detail = f": {last.error}" if last.error else ""
                        raise AllAttemptsFailed(
                            f"No eligible fallback model remains after {last.status} on "
                            f"{last.model_id}{detail}", history
                        )
                    raise

            model, reservation = selection
            attempted_models.add(model.id)
            output_limit = max_output_tokens if max_output_tokens is not None else model.max_output_tokens
            if output_limit < 1:
                reservation.release()
                raise ValueError("max_output_tokens must be at least 1")
            worst_case_cost = _estimated_cost_usd(model, estimated_input, output_limit)
            if max_cost_usd is not None and call_spend_usd + worst_case_cost > max_cost_usd + 1e-12:
                reservation.release()
                error = BudgetExceeded(
                    f"Estimated call cost ${call_spend_usd + worst_case_cost:.6f} exceeds "
                    f"max_cost_usd=${max_cost_usd:.6f}"
                )
                self._record_attempt(
                    request_id=request_id, task=task, caller=caller, model=model,
                    attempt_no=attempt_no, status="budget_blocked", latency_ms=0,
                    prompt_hash=prompt_hash, http_status=None, error_message=str(error),
                )
                raise error
            budget_reservation_id = str(uuid.uuid4())
            try:
                spent, cap, soft_warn_pct = self.store.reserve_budget(
                    budget_reservation_id, request_id, route.budget_group,
                    worst_case_cost, int(time.time() + request_timeout + 60),
                )
            except BudgetExceeded as exc:
                reservation.release()
                self._record_attempt(
                    request_id=request_id, task=task, caller=caller, model=model,
                    attempt_no=attempt_no, status="budget_blocked", latency_ms=0,
                    prompt_hash=prompt_hash, http_status=None, error_message=str(exc),
                )
                raise
            except Exception:
                reservation.release()
                raise
            if cap > 0 and (spent + worst_case_cost) * 100 / cap >= soft_warn_pct:
                self._emit_log(
                    "budget_soft_warning", request_id=request_id, task=task,
                    budget_group=route.budget_group, spent_usd=spent,
                    reserved_usd=worst_case_cost, cap_usd=cap,
                )
            request = NormalizedRequest(
                messages=current_messages,
                temperature=request_temperature,
                max_output_tokens=output_limit if output_limit > 0 else None,
                request_id=request_id,
                timeout_s=min(
                    request_timeout,
                    max(0.001, logical_deadline - time.monotonic()),
                ),
                json_mode=structured_schema is not None and model.supports_json_mode,
                response_format=(
                    {"type": "json_object"}
                    if structured_schema is not None and model.supports_json_mode else None
                ),
            )
            adapter = (_adapter_overrides or self.adapters).get(model.provider)
            if adapter is None:
                reservation.release()
                self.store.release_budget(budget_reservation_id)
                raise ConfigError(f"No adapter registered for provider {model.provider!r}")

            self._emit_log(
                "api_call_started",
                request_id=request_id,
                task=task,
                caller=caller,
                attempt_no=attempt_no,
                model_id=model.id,
                provider=model.provider,
                prompt_hash=prompt_hash,
            )
            started = time.monotonic()
            try:
                response = adapter.complete(request, model)
            except Exception as exc:
                secret_values = [os.getenv(model.api_key_env, "")]
                try:
                    error_class = adapter.classify_error(exc)
                except Exception:
                    # A faulty custom adapter must not strand either local
                    # concurrency capacity or a durable budget reservation.
                    reservation.release()
                    self.store.release_budget(budget_reservation_id)
                    raise
                provider_error = exc if isinstance(exc, ProviderError) else ProviderError(
                    _safe_error(exc, secret_values), error_class
                )
                if provider_error.used_tokens:
                    reservation.settle(estimated_input, output_limit, worst_case_cost)
                    call_spend_usd += worst_case_cost
                else:
                    reservation.release()
                    self.store.release_budget(budget_reservation_id)
                history.append(Attempt(
                    model.id,
                    attempt_no,
                    error_class.value,
                    error=_safe_error(provider_error, secret_values),
                ))
                latency_ms = int((time.monotonic() - started) * 1000)
                self._record_attempt(
                    request_id=request_id, task=task, caller=caller, model=model,
                    attempt_no=attempt_no, status=error_class.value, latency_ms=latency_ms,
                    prompt_hash=prompt_hash, http_status=provider_error.http_status,
                    error_message=_safe_error(provider_error, secret_values),
                    input_tokens=estimated_input if provider_error.used_tokens else None,
                    output_tokens=output_limit if provider_error.used_tokens else None,
                    cost_usd=worst_case_cost if provider_error.used_tokens else 0.0,
                    budget_reservation_id=(budget_reservation_id if provider_error.used_tokens else None),
                )
                failures = self._consecutive_failures.get(model.id, 0) + 1
                self._consecutive_failures[model.id] = failures

                if error_class == ErrorClass.RATE_LIMITED:
                    delay = provider_error.retry_after_s
                    cooldown_s = max(1, math.ceil(delay if delay is not None else 60))
                    self.limiter.set_cooldown(model.id, int(time.time()) + cooldown_s, "rate limited",
                                              consecutive_failures=failures)
                elif error_class == ErrorClass.AUTH_ERROR:
                    self._disabled_models.add(model.id)
                elif failures >= 3 and error_class in (ErrorClass.SERVER_ERROR, ErrorClass.TIMEOUT):
                    self.limiter.set_cooldown(model.id, int(time.time()) + 60, error_class.value,
                                              consecutive_failures=failures)

                if error_class == ErrorClass.SERVER_ERROR and model.id not in retried_server_models:
                    retried_server_models.add(model.id)
                    attempted_models.discard(model.id)
                    retry_model = model
                    retry_messages = current_messages
                    backoff_cap = min(60.0, 2.0 ** min(attempt_no - 1, 6))
                    remaining = logical_deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(random.random() * backoff_cap, remaining))
                elif error_class == ErrorClass.CLIENT_ERROR and not _is_context_error(provider_error):
                    raise AllAttemptsFailed("Provider rejected the request", history) from exc
                # Other classes move to the next candidate. The logical attempt cap
                # applies across retries and fallback models.
                continue

            cost = _cost_usd(model, response)
            validated_data = None
            if structured_schema is not None:
                try:
                    validated_data = validate_json_output(
                        response.text, structured_schema, custom_validator
                    )
                except OutputValidationError as validation_error:
                    last_validation_error = validation_error
                    validation_failures_by_tier[model.tier] = (
                        validation_failures_by_tier.get(model.tier, 0) + 1
                    )
                    reservation.settle(response.input_tokens, response.output_tokens, cost)
                    call_spend_usd += cost
                    history.append(Attempt(
                        model.id, attempt_no, "invalid_output",
                        error="; ".join(validation_error.problems)[:500], response=response,
                    ))
                    latency_ms = int((time.monotonic() - started) * 1000)
                    self._record_attempt(
                        request_id=request_id, task=task, caller=caller, model=model,
                        attempt_no=attempt_no, status="invalid_output", latency_ms=latency_ms,
                        prompt_hash=prompt_hash, http_status=None,
                        error_message=(
                            f"Structured output validation failed "
                            f"({len(validation_error.problems)} issue(s))"
                        ),
                        input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                        cost_usd=cost,
                        budget_reservation_id=budget_reservation_id,
                    )
                    if model.id not in validation_retried_models:
                        validation_retried_models.add(model.id)
                        attempted_models.discard(model.id)
                        retry_model = model
                        retry_messages = current_messages + [
                            Message(role="assistant", content=response.text),
                            Message(
                                role="user",
                                content=(
                                    "The previous response did not pass JSON validation. "
                                    "Correct these issues and return only valid JSON: "
                                    + "; ".join(validation_error.problems)
                                ),
                            ),
                        ]
                    elif validation_failures_by_tier[model.tier] >= 2:
                        escalated_tiers.add(model.tier)
                    continue
                except Exception:
                    # A user validator may itself raise; settle before propagating it.
                    reservation.settle(response.input_tokens, response.output_tokens, cost)
                    call_spend_usd += cost
                    latency_ms = int((time.monotonic() - started) * 1000)
                    self._record_attempt(
                        request_id=request_id, task=task, caller=caller, model=model,
                        attempt_no=attempt_no, status="validator_error", latency_ms=latency_ms,
                        prompt_hash=prompt_hash, http_status=None,
                        error_message="Custom output validator raised an exception",
                        input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                        cost_usd=cost, budget_reservation_id=budget_reservation_id,
                    )
                    raise

            reservation.settle(response.input_tokens, response.output_tokens, cost)
            call_spend_usd += cost
            self.limiter.clear_cooldown(model.id)
            self._consecutive_failures[model.id] = 0
            history.append(Attempt(model.id, attempt_no, "ok", response=response))
            latency_ms = int((time.monotonic() - started) * 1000)
            self._record_attempt(
                request_id=request_id, task=task, caller=caller, model=model,
                attempt_no=attempt_no, status="ok", latency_ms=latency_ms,
                prompt_hash=prompt_hash, http_status=None,
                input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                cost_usd=cost,
                budget_reservation_id=budget_reservation_id,
            )
            if cache_enabled:
                self.store.cache_put(
                    cache_key, task, model_family,
                    {
                        "text": response.text, "model_id": model.id,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    },
                    ttl,
                )
            return CallResult(
                text=response.text,
                model_id=model.id,
                cost_usd=cost,
                cached=False,
                attempts=history,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                request_id=request_id,
                data=validated_data,
                comparison_cost_usd=_comparison_cost_usd(
                    model, response.input_tokens, response.output_tokens,
                ),
                savings_usd=_savings_usd(
                    model, response.input_tokens, response.output_tokens, cost,
                ),
            )

        if last_validation_error is not None and history and history[-1].status == "invalid_output":
            raise OutputValidationError(
                str(last_validation_error), last_validation_error.problems, history
            )
        raise AllAttemptsFailed("All configured model attempts failed", history)

    def status(self) -> list[dict[str, object]]:
        return self.limiter.status(self.config.models)

    def cache_clear(self, task: str | None = None) -> int:
        """Delete cached responses, optionally limited to one task; return count."""
        return self.store.cache_clear(task)

    def report(self, month: str | None = None) -> dict[str, object]:
        """Return actual spend by budget group, task, and model for a UTC month."""
        if month is None:
            month = time.strftime("%Y-%m", time.gmtime())
        if len(month) != 7 or month[4] != "-" or not month[:4].isdigit() or not month[5:].isdigit():
            raise ValueError("month must use YYYY-MM format")
        start = calendar.timegm(time.strptime(month + "-01", "%Y-%m-%d"))
        end_struct = time.strptime(month + "-01", "%Y-%m-%d")
        end = calendar.timegm((end_struct.tm_year + (end_struct.tm_mon == 12),
                               (end_struct.tm_mon % 12) + 1, 1, 0, 0, 0, 0, 0, 0))
        return self.store.budget_report(month, start, end)

    def _record_attempt(
        self,
        *,
        request_id: str,
        task: str,
        caller: str | None,
        model: ModelConfig,
        attempt_no: int,
        status: str,
        latency_ms: int,
        prompt_hash: str,
        http_status: int | None,
        error_message: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float = 0.0,
        budget_reservation_id: str | None = None,
    ) -> None:
        ts = int(time.time())
        entry = CallLogEntry(
            id=str(uuid.uuid4()), ts=ts, request_id=request_id, task=task, caller=caller,
            model_id=model.id, attempt_no=attempt_no, status=status, http_status=http_status,
            latency_ms=latency_ms, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd, shadow_cost_usd=_shadow_cost_usd(model, input_tokens, output_tokens),
            error_message=error_message, prompt_hash=prompt_hash,
        )
        if budget_reservation_id is not None:
            self.store.settle_budget_and_log(budget_reservation_id, entry)
        else:
            self.store.log_call(entry)
        self._emit_log(
            "api_call_finished", request_id=request_id, task=task, caller=caller,
            attempt_no=attempt_no, model_id=model.id, provider=model.provider, status=status,
            http_status=http_status, latency_ms=latency_ms, input_tokens=input_tokens,
            output_tokens=output_tokens, cost_usd=cost_usd, error=error_message,
            prompt_hash=prompt_hash,
        )

    def _record_cache_hit(
        self, *, request_id: str, task: str, caller: str | None,
        model_id: str, prompt_hash: str,
    ) -> None:
        self.store.log_call(CallLogEntry(
            id=str(uuid.uuid4()), ts=int(time.time()), request_id=request_id,
            task=task, caller=caller, model_id=model_id, attempt_no=0,
            status="cache_hit", http_status=None, latency_ms=0,
            input_tokens=0, output_tokens=0, cost_usd=0.0, shadow_cost_usd=0.0,
            error_message=None, prompt_hash=prompt_hash,
        ))
        self._emit_log(
            "cache_hit", request_id=request_id, task=task, caller=caller,
            model_id=model_id, prompt_hash=prompt_hash,
        )

    def _emit_log(self, event: str, **fields: object) -> None:
        """Keep an unavailable activity log from leaking provider capacity.

        The SQLite attempt record remains authoritative. A full or temporarily
        unavailable JSONL log must not leave quota/budget reservations active or
        turn a completed provider call into an apparent failure.
        """
        try:
            self.call_logger.emit(event, **fields)
        except OSError:
            pass

    def close(self) -> None:
        self._async_executor.shutdown(wait=True, cancel_futures=True)
        seen: set[int] = set()
        for adapter in self.adapters.values():
            closer = getattr(adapter, "close", None)
            if closer is not None and id(adapter) not in seen:
                closer()
                seen.add(id(adapter))
        self.store.close()

    def __enter__(self) -> "Router":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _estimate_tokens(messages: Sequence[Message]) -> int:
    text = "\n".join(message.content for message in messages)
    return max(1, math.ceil(len(text) / 4))


def _prompt_hash(messages: Sequence[Message]) -> str:
    payload = [item.model_dump() for item in messages]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _model_family(models: Sequence[ModelConfig], tier_order: Sequence[str]) -> str:
    ordered = []
    for tier in tier_order:
        tier_models = sorted(
            (model for model in models if model.tier == tier),
            key=lambda model: (model.priority, model.id),
        )
        ordered.extend({
            "id": model.id, "provider": model.provider, "name": model.model_name,
            # A cache entry must not survive a provider disablement or a
            # privacy-policy/capability change in the active configuration.
            "base_url": model.base_url, "api_key_env": model.api_key_env,
            "enabled": model.enabled, "tier": model.tier,
            "data_policy": model.data_policy,
            "context": model.context_window, "max_output": model.max_output_tokens,
            "json_mode": model.supports_json_mode, "tools": model.supports_tools,
            "vision": model.supports_vision,
            "input_price": model.input_price_per_mtok,
            "output_price": model.output_price_per_mtok,
        } for model in tier_models)
    return json.dumps(ordered, sort_keys=True, separators=(",", ":"))


def _cache_key(
    task: str,
    messages: Sequence[Message],
    schema: JSONSchema | None,
    temperature: float,
    max_output_tokens: int | None,
    model_family: str,
    sensitivity: str | None,
) -> str:
    schema_json = None if schema is None else schema_as_json_schema(schema)
    payload = {
        "task": task,
        "messages": [item.model_dump() for item in messages],
        "schema": schema_json,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "model_family": model_family,
        "sensitivity": sensitivity,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _cost_usd(model: ModelConfig, response: NormalizedResponse) -> float:
    if model.is_free_tier:
        return 0.0
    return (
        response.input_tokens * model.input_price_per_mtok
        + response.output_tokens * model.output_price_per_mtok
    ) / 1_000_000


def _estimated_cost_usd(model: ModelConfig, input_tokens: int, output_tokens: int) -> float:
    if model.is_free_tier:
        return 0.0
    return (
        input_tokens * model.input_price_per_mtok
        + output_tokens * model.output_price_per_mtok
    ) / 1_000_000


def _comparison_cost_usd(
    model: ModelConfig, input_tokens: int, output_tokens: int,
) -> float | None:
    """Price usage at configured shadow rates, or return None without a full baseline."""
    input_price = model.shadow_input_price_per_mtok
    output_price = model.shadow_output_price_per_mtok
    if input_price is None or output_price is None:
        return None
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def _savings_usd(
    model: ModelConfig, input_tokens: int, output_tokens: int, cost_usd: float,
) -> float | None:
    comparison = _comparison_cost_usd(model, input_tokens, output_tokens)
    return None if comparison is None else comparison - cost_usd


def _shadow_cost_usd(
    model: ModelConfig, input_tokens: int | None, output_tokens: int | None
) -> float:
    if input_tokens is None or output_tokens is None:
        return 0.0
    input_price = model.shadow_input_price_per_mtok
    output_price = model.shadow_output_price_per_mtok
    if input_price is None or output_price is None:
        return 0.0
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def _safe_error(exc: Exception, secrets: Sequence[str] = ()) -> str:
    message = str(exc)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    for marker in ("sk-", "Bearer ", "key="):
        position = message.lower().find(marker.lower())
        if position >= 0:
            message = message[:position] + "[redacted]"
    return message[:500]


def _is_context_error(error: ProviderError) -> bool:
    text = str(error).lower()
    return any(phrase in text for phrase in ("context length", "context window", "too many tokens", "maximum context"))
