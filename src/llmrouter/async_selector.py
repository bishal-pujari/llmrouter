"""Async counterpart to the deterministic model selector."""

from __future__ import annotations

from dataclasses import dataclass

from llmrouter.async_limiter import AsyncLimitReservation, AsyncRateLimiter
from llmrouter.config import ModelConfig, TaskRoute
from llmrouter.errors import NoModelAvailable


@dataclass
class AsyncSelection:
    model: ModelConfig
    reservation: AsyncLimitReservation


class AsyncModelSelector:
    def __init__(self, models: list[ModelConfig], limiter: AsyncRateLimiter) -> None:
        self.models = models
        self.limiter = limiter

    async def select_and_reserve(
        self, route: TaskRoute, *, estimated_input_tokens: int, max_output_tokens: int | None,
        sensitivity: str | None = None, excluded_model_ids: set[str] | None = None,
        excluded_tiers: set[str] | None = None, allowed_model_ids: set[str] | None = None,
    ) -> AsyncSelection:
        required_sensitivity = sensitivity or route.sensitivity
        eligible: list[tuple[ModelConfig, float]] = []
        for model in self.models:
            total = estimated_input_tokens + (model.max_output_tokens if max_output_tokens is None else max_output_tokens)
            if not (
                model.enabled
                and (allowed_model_ids is None or model.id in allowed_model_ids)
                and model.id not in (excluded_model_ids or set())
                and model.tier not in (excluded_tiers or set())
                and model.tier in route.tier_order
                and total <= model.context_window
                and (max_output_tokens is None or max_output_tokens <= model.max_output_tokens)
                and self._supports(model, route, total)
                and self._sensitivity_allows(model, required_sensitivity)
                and not await self.limiter.is_cooling_down(model.id)
            ):
                continue
            eligible.append((model, await self.limiter.headroom_score(model)))
        tier_rank = {tier: index for index, tier in enumerate(route.tier_order)}
        eligible.sort(key=lambda item: (tier_rank[item[0].tier], item[0].priority, -item[1], item[0].id))
        for model, _ in eligible:
            output_reservation = model.max_output_tokens if max_output_tokens is None else max_output_tokens
            reservation = await self.limiter.reserve(model, estimated_input_tokens, output_reservation)
            if reservation is not None:
                return AsyncSelection(model, reservation)
        raise NoModelAvailable("No eligible model has capacity for task route")

    @staticmethod
    def _supports(model: ModelConfig, route: TaskRoute, total_tokens: int) -> bool:
        if total_tokens > model.context_window:
            return False
        capability_fields = {"json_mode": "supports_json_mode", "tools": "supports_tools", "vision": "supports_vision"}
        for requirement, field in capability_fields.items():
            if route.requires.get(requirement, False) and not getattr(model, field):
                return False
        minimum = route.requires.get("min_context", 0)
        return isinstance(minimum, int) and minimum >= 0 and model.context_window >= minimum

    @staticmethod
    def _sensitivity_allows(model: ModelConfig, sensitivity: str) -> bool:
        return sensitivity == "public" or (sensitivity == "private" and model.data_policy == "no_train")
