"""Async Router implementation for Cloudflare Workers and D1.

This is deliberately separate from ``Router``. The local implementation is
thread-safe and synchronous; Workers run on an async Pyodide isolate where
threading and durable local files are unavailable.
"""

from __future__ import annotations

import asyncio
import calendar
import math
import os
import random
import time
import uuid
from typing import Any, Callable, Sequence

from llmrouter.adapters.anthropic import AnthropicAdapter
from llmrouter.adapters.base import ErrorClass, Message, NormalizedRequest, ProviderAdapter, ProviderError
from llmrouter.adapters.cloudflare import CloudflareAdapter
from llmrouter.adapters.gemini import GeminiAdapter
from llmrouter.adapters.openai_compat import OpenAICompatAdapter
from llmrouter.async_limiter import AsyncRateLimiter
from llmrouter.async_selector import AsyncModelSelector
from llmrouter.config import ModelConfig, RouterConfig, load_config_text
from llmrouter.errors import BudgetExceeded, ConfigError, NoModelAvailable, OutputValidationError
from llmrouter.executor import AllAttemptsFailed, Attempt
from llmrouter.router import (
    BatchResult,
    CallResult,
    _cache_key,
    _comparison_cost_usd,
    _cost_usd,
    _estimated_cost_usd,
    _is_context_error,
    _model_family,
    _prompt_hash,
    _safe_error,
    _savings_usd,
    _shadow_cost_usd,
    _estimate_tokens,
)
from llmrouter.store.async_protocol import AsyncStore
from llmrouter.store.d1 import D1Store
from llmrouter.store.protocol import CallLogEntry
from llmrouter.validation import CustomValidator, JSONSchema, json_instruction, schema_as_json_schema, validate_json_output


class AsyncRouter:
    """D1-backed async router with no filesystem or thread-pool dependency."""

    def __init__(self, config: RouterConfig, store: AsyncStore,
                 *, adapters: dict[str, ProviderAdapter] | None = None,
                 secret_resolver: Callable[[str], str | None] | None = None) -> None:
        self.config = config
        self.store = store
        self._secret_resolver = secret_resolver or os.environ.get
        self.limiter = AsyncRateLimiter(store, config.settings.limit_safety_fraction)
        self.selector = AsyncModelSelector(config.models, self.limiter)
        self.adapters: dict[str, ProviderAdapter] = adapters or {
            "openai_compat": OpenAICompatAdapter(credential_resolver=self._secret_resolver),
            "anthropic": AnthropicAdapter(credential_resolver=self._secret_resolver),
            "gemini": GeminiAdapter(credential_resolver=self._secret_resolver),
            "cloudflare": CloudflareAdapter(credential_resolver=self._secret_resolver),
        }
        self._disabled_models: set[str] = set()
        self._consecutive_failures: dict[str, int] = {}

    @classmethod
    async def from_config_text(cls, text: str, database: Any, *, format: str = "json",
                               secret_resolver: Callable[[str], str | None] | None = None) -> "AsyncRouter":
        config = load_config_text(text, format=format)
        store = D1Store(database)
        await store.migrate()
        await store.upsert_config(config)
        return cls(config, store, secret_resolver=secret_resolver)

    async def acall(self, task: str, messages: Sequence[dict[str, str] | Message], **options: Any) -> CallResult:
        return await self._execute(task, messages, **options)

    async def acall_json(self, task: str, messages: Sequence[dict[str, str] | Message], schema: JSONSchema, **options: Any) -> CallResult:
        schema_json = schema_as_json_schema(schema)
        options["system"] = json_instruction(schema_json, options.get("system"))
        return await self._execute(task, messages, structured_schema=schema, **options)

    async def batch(self, items: Sequence[Sequence[dict[str, str] | Message] | dict[str, Any]], *, task: str,
                    concurrency: int = 8, **call_options: Any) -> list[BatchResult]:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        gate = asyncio.Semaphore(concurrency)
        results: list[BatchResult | None] = [None] * len(items)

        async def run_one(index: int, item: Sequence[dict[str, str] | Message] | dict[str, Any]) -> None:
            async with gate:
                try:
                    if isinstance(item, dict):
                        if "messages" not in item:
                            raise ValueError("batch mapping items must include 'messages'")
                        outcome = await self.acall(task, item["messages"], **{**call_options, **item.get("options", {})})
                    else:
                        outcome = await self.acall(task, item, **call_options)
                    results[index] = BatchResult(index=index, result=outcome)
                except Exception as exc:
                    results[index] = BatchResult(index=index, error=exc)

        await asyncio.gather(*(run_one(index, item) for index, item in enumerate(items)))
        return [item for item in results if item is not None]

    async def _execute(
        self, task: str, messages: Sequence[dict[str, str] | Message], *, caller: str | None = None,
        system: str | None = None, temperature: float | None = None, max_output_tokens: int | None = None,
        sensitivity: str | None = None, timeout_s: float | None = None, force_model: str | None = None,
        use_cache: bool = True, cache_ttl_seconds: int | None = None, max_cost_usd: float | None = None,
        structured_schema: JSONSchema | None = None, custom_validator: CustomValidator | None = None,
    ) -> CallResult:
        route = self.config.task_routes.get(task)
        if route is None:
            raise ConfigError(f"Unknown task route {task!r}")
        if not messages:
            raise ValueError("messages must contain at least one message")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        if timeout_s is not None and (timeout_s <= 0 or not math.isfinite(timeout_s)):
            raise ValueError("timeout_s must be greater than 0")
        if max_cost_usd is not None and (max_cost_usd < 0 or not math.isfinite(max_cost_usd)):
            raise ValueError("max_cost_usd cannot be negative")
        normalized = [Message.model_validate(item) for item in messages]
        if system:
            normalized.insert(0, Message(role="system", content=system))
        request_id = str(uuid.uuid4())
        request_timeout = self.config.settings.default_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + request_timeout * route.max_attempts
        request_temperature = route.default_temperature if temperature is None else temperature
        if request_temperature < 0 or not math.isfinite(request_temperature):
            raise ValueError("temperature must be finite and non-negative")
        required_sensitivity = sensitivity or route.sensitivity
        ttl = route.cache_ttl_seconds if cache_ttl_seconds is None else cache_ttl_seconds
        cache_enabled = use_cache and self.config.settings.cache_enabled and ttl != 0 and custom_validator is None
        cache_models = self.config.models if force_model is None else [m for m in self.config.models if m.id == force_model]
        if force_model is not None and not cache_models:
            raise ConfigError(f"Unknown model {force_model!r}")
        model_family = _model_family(cache_models, route.tier_order)
        cache_key = _cache_key(task, normalized, structured_schema, request_temperature, max_output_tokens, model_family, required_sensitivity)
        prompt_hash = _prompt_hash(normalized)
        if cache_enabled:
            cached = await self.store.cache_get(cache_key)
            if cached is not None:
                cached_text = str(cached["text"])
                data = validate_json_output(cached_text, structured_schema) if structured_schema is not None else None
                model_id = str(cached["model_id"])
                await self._record_cache_hit(request_id, task, caller, model_id, prompt_hash)
                return CallResult(cached_text, model_id, 0.0, True, [Attempt(model_id, 0, "cache_hit")], int(cached["input_tokens"]), int(cached["output_tokens"]), request_id, data=data)

        history: list[Attempt] = []
        attempted: set[str] = set()
        retried: set[str] = set()
        retry_model: ModelConfig | None = None
        retry_messages: list[Message] | None = None
        spent_usd = 0.0
        last_validation: OutputValidationError | None = None
        validation_retried: set[str] = set()
        escalated_tiers: set[str] = set()

        for attempt_no in range(1, route.max_attempts + 1):
            current = retry_messages if retry_messages is not None else normalized
            retry_messages = None
            estimated_input = _estimate_tokens(current)
            selection = None
            if retry_model is not None:
                retry_candidate = retry_model
                output = max_output_tokens if max_output_tokens is not None else retry_model.max_output_tokens
                reservation = await self.limiter.reserve(retry_model, estimated_input, output)
                retry_model = None
                if reservation is not None:
                    selection = (retry_candidate, reservation)
            if selection is None:
                try:
                    selected = await self.selector.select_and_reserve(
                        route, estimated_input_tokens=estimated_input, max_output_tokens=max_output_tokens,
                        sensitivity=sensitivity, excluded_model_ids=attempted | self._disabled_models,
                        excluded_tiers=escalated_tiers,
                        allowed_model_ids={force_model} if force_model is not None else None,
                    )
                    selection = (selected.model, selected.reservation)
                except NoModelAvailable:
                    if history:
                        if last_validation is not None and history[-1].status == "invalid_output":
                            raise OutputValidationError(str(last_validation), last_validation.problems, history)
                        last = history[-1]
                        raise AllAttemptsFailed(f"No eligible fallback model remains after {last.status} on {last.model_id}", history)
                    raise
            model, reservation = selection
            attempted.add(model.id)
            output_limit = max_output_tokens if max_output_tokens is not None else model.max_output_tokens
            if output_limit < 1:
                await reservation.release()
                raise ValueError("max_output_tokens must be at least 1")
            worst_case = _estimated_cost_usd(model, estimated_input, output_limit)
            if max_cost_usd is not None and spent_usd + worst_case > max_cost_usd + 1e-12:
                await reservation.release()
                error = BudgetExceeded(f"Estimated call cost exceeds max_cost_usd=${max_cost_usd:.6f}")
                await self._record_attempt(request_id, task, caller, model, attempt_no, "budget_blocked", 0, _prompt_hash(current), None, str(error))
                raise error
            budget_id = str(uuid.uuid4())
            try:
                await self.store.reserve_budget(budget_id, request_id, route.budget_group, worst_case, int(time.time() + request_timeout + 60))
            except Exception:
                await reservation.release()
                raise
            request = NormalizedRequest(
                messages=current, temperature=request_temperature, max_output_tokens=output_limit,
                request_id=request_id, timeout_s=min(request_timeout, max(0.001, deadline - time.monotonic())),
                json_mode=structured_schema is not None and model.supports_json_mode,
                response_format={"type": "json_object"} if structured_schema is not None and model.supports_json_mode else None,
            )
            adapter = self.adapters.get(model.provider)
            if adapter is None:
                await reservation.release(); await self.store.release_budget(budget_id)
                raise ConfigError(f"No adapter registered for provider {model.provider!r}")
            started = time.monotonic()
            try:
                response = await adapter.acomplete(request, model)
            except Exception as exc:
                try:
                    error_class = adapter.classify_error(exc)
                except Exception:
                    await reservation.release()
                    await self.store.release_budget(budget_id)
                    raise
                safe = _safe_error(exc, [self._secret_resolver(model.api_key_env) or ""])
                provider_error = exc if isinstance(exc, ProviderError) else ProviderError(safe, error_class)
                if provider_error.used_tokens:
                    await reservation.settle(estimated_input, output_limit, worst_case)
                    spent_usd += worst_case
                    await self._record_attempt(request_id, task, caller, model, attempt_no, error_class.value, int((time.monotonic()-started)*1000), _prompt_hash(current), provider_error.http_status, safe, estimated_input, output_limit, worst_case, budget_id)
                else:
                    await reservation.release(); await self.store.release_budget(budget_id)
                    await self._record_attempt(request_id, task, caller, model, attempt_no, error_class.value, int((time.monotonic()-started)*1000), _prompt_hash(current), provider_error.http_status, safe)
                history.append(Attempt(model.id, attempt_no, error_class.value, error=safe))
                failures = self._consecutive_failures.get(model.id, 0) + 1
                self._consecutive_failures[model.id] = failures
                if error_class == ErrorClass.RATE_LIMITED:
                    await self.limiter.set_cooldown(model.id, int(time.time()) + max(1, math.ceil(provider_error.retry_after_s or 60)), "rate limited", consecutive_failures=failures)
                elif error_class == ErrorClass.AUTH_ERROR:
                    self._disabled_models.add(model.id)
                elif failures >= 3 and error_class in (ErrorClass.SERVER_ERROR, ErrorClass.TIMEOUT):
                    await self.limiter.set_cooldown(model.id, int(time.time()) + 60, error_class.value, consecutive_failures=failures)
                if error_class == ErrorClass.SERVER_ERROR and model.id not in retried:
                    retried.add(model.id); attempted.discard(model.id); retry_model = model; retry_messages = current
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(min(random.random() * min(60.0, 2.0 ** min(attempt_no - 1, 6)), remaining))
                elif error_class == ErrorClass.CLIENT_ERROR and not _is_context_error(provider_error):
                    raise AllAttemptsFailed("Provider rejected the request", history) from exc
                continue

            cost = _cost_usd(model, response)
            data = None
            if structured_schema is not None:
                try:
                    data = validate_json_output(response.text, structured_schema, custom_validator)
                except OutputValidationError as exc:
                    last_validation = exc
                    await reservation.settle(response.input_tokens, response.output_tokens, cost); spent_usd += cost
                    await self._record_attempt(request_id, task, caller, model, attempt_no, "invalid_output", int((time.monotonic()-started)*1000), _prompt_hash(current), None, "Structured output validation failed", response.input_tokens, response.output_tokens, cost, budget_id)
                    history.append(Attempt(model.id, attempt_no, "invalid_output", error="; ".join(exc.problems)[:500], response=response))
                    if model.id not in validation_retried:
                        validation_retried.add(model.id); attempted.discard(model.id); retry_model = model
                        retry_messages = current + [Message(role="assistant", content=response.text), Message(role="user", content="Return only valid JSON and correct the validation issues: " + "; ".join(exc.problems))]
                    else:
                        escalated_tiers.add(model.tier)
                    continue
                except Exception:
                    await reservation.settle(response.input_tokens, response.output_tokens, cost)
                    spent_usd += cost
                    await self._record_attempt(
                        request_id, task, caller, model, attempt_no, "validator_error",
                        int((time.monotonic() - started) * 1000), _prompt_hash(current), None,
                        "Custom output validator raised an exception", response.input_tokens,
                        response.output_tokens, cost, budget_id,
                    )
                    raise
            await reservation.settle(response.input_tokens, response.output_tokens, cost); spent_usd += cost
            await self.limiter.clear_cooldown(model.id)
            self._consecutive_failures[model.id] = 0
            history.append(Attempt(model.id, attempt_no, "ok", response=response))
            await self._record_attempt(request_id, task, caller, model, attempt_no, "ok", int((time.monotonic()-started)*1000), _prompt_hash(current), None, None, response.input_tokens, response.output_tokens, cost, budget_id)
            if cache_enabled:
                await self.store.cache_put(cache_key, task, model_family, {"text": response.text, "model_id": model.id, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens}, ttl)
            return CallResult(response.text, model.id, cost, False, history, response.input_tokens, response.output_tokens, request_id, data=data, comparison_cost_usd=_comparison_cost_usd(model, response.input_tokens, response.output_tokens), savings_usd=_savings_usd(model, response.input_tokens, response.output_tokens, cost))
        if last_validation is not None:
            raise OutputValidationError(str(last_validation), last_validation.problems, history)
        raise AllAttemptsFailed("All configured model attempts failed", history)

    async def _record_attempt(self, request_id: str, task: str, caller: str | None, model: ModelConfig,
                              attempt_no: int, status: str, latency_ms: int, prompt_hash: str,
                              http_status: int | None, error_message: str | None,
                              input_tokens: int | None = None, output_tokens: int | None = None,
                              cost_usd: float = 0.0, budget_reservation_id: str | None = None) -> None:
        entry = CallLogEntry(str(uuid.uuid4()), int(time.time()), request_id, task, caller, model.id, attempt_no, status, http_status, latency_ms, input_tokens, output_tokens, cost_usd, _shadow_cost_usd(model, input_tokens, output_tokens), error_message, prompt_hash)
        if budget_reservation_id is None:
            await self.store.log_call(entry)
        else:
            await self.store.settle_budget_and_log(budget_reservation_id, entry)

    async def _record_cache_hit(self, request_id: str, task: str, caller: str | None, model_id: str, prompt_hash: str) -> None:
        await self.store.log_call(CallLogEntry(str(uuid.uuid4()), int(time.time()), request_id, task, caller, model_id, 0, "cache_hit", None, 0, 0, 0, 0.0, 0.0, None, prompt_hash))

    async def status(self) -> list[dict[str, object]]:
        return await self.limiter.status(self.config.models)

    async def cache_clear(self, task: str | None = None) -> int:
        return await self.store.cache_clear(task)

    async def report(self, month: str | None = None) -> dict[str, object]:
        month = month or time.strftime("%Y-%m", time.gmtime())
        if len(month) != 7 or month[4] != "-":
            raise ValueError("month must use YYYY-MM format")
        start = calendar.timegm(time.strptime(month + "-01", "%Y-%m-%d"))
        year, month_number = int(month[:4]), int(month[5:])
        end = calendar.timegm((year + (month_number == 12), (month_number % 12) + 1, 1, 0, 0, 0, 0, 0, 0))
        return await self.store.budget_report(month, start, end)

    async def close(self) -> None:
        await self.store.close()
