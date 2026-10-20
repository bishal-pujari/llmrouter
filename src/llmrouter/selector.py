"""Deterministic task-based model selection and quota reservation."""

from __future__ import annotations

from dataclasses import dataclass

from llmrouter.config import ModelConfig, TaskRoute
from llmrouter.errors import NoModelAvailable
from llmrouter.limiter import LimitReservation, RateLimiter


@dataclass
class Selection:
    model: ModelConfig
    reservation: LimitReservation


class ModelSelector:
    def __init__(self, models: list[ModelConfig], limiter: RateLimiter) -> None:
        self.models = models
        self.limiter = limiter

    def select_and_reserve(
        self,
        route: TaskRoute,
        *,
        estimated_input_tokens: int,
        max_output_tokens: int | None,
        sensitivity: str | None = None,
        excluded_model_ids: set[str] | None = None,
        excluded_tiers: set[str] | None = None,
        allowed_model_ids: set[str] | None = None,
        now: int | None = None,
    ) -> Selection:
        """Pick the first eligible model by tier, priority, then quota headroom."""
        required_sensitivity = sensitivity or route.sensitivity
        eligible = [
            model for model in self.models
            if model.enabled
            and (allowed_model_ids is None or model.id in allowed_model_ids)
            and model.id not in (excluded_model_ids or set())
            and model.tier not in (excluded_tiers or set())
            and model.tier in route.tier_order
            and self._supports(
                model, route,
                estimated_input_tokens + (
                    model.max_output_tokens if max_output_tokens is None else max_output_tokens
                ),
            )
            and (max_output_tokens is None or max_output_tokens <= model.max_output_tokens)
            and self._sensitivity_allows(model, required_sensitivity)
            and not self.limiter.is_cooling_down(model.id, now=now)
        ]
        tier_rank = {tier: index for index, tier in enumerate(route.tier_order)}
        eligible.sort(key=lambda model: (
            tier_rank[model.tier], model.priority,
            -self.limiter.headroom_score(model, now=now), model.id,
        ))

        for model in eligible:
            output_reservation = (
                model.max_output_tokens if max_output_tokens is None else max_output_tokens
            )
            reservation = self.limiter.reserve(
                model,
                estimated_input_tokens,
                output_reservation,
                now=now,
            )
            if reservation is not None:
                return Selection(model=model, reservation=reservation)
        raise NoModelAvailable(f"No eligible model has capacity for task route")

    @staticmethod
    def _supports(model: ModelConfig, route: TaskRoute, total_tokens: int) -> bool:
        if total_tokens > model.context_window:
            return False
        requirements = route.requires
        capability_fields = {
            "json_mode": "supports_json_mode",
            "tools": "supports_tools",
            "vision": "supports_vision",
        }
        for requirement, model_field in capability_fields.items():
            if requirements.get(requirement, False) and not getattr(model, model_field):
                return False
        min_context = requirements.get("min_context", 0)
        if not isinstance(min_context, int) or min_context < 0:
            return False
        return model.context_window >= min_context

    @staticmethod
    def _sensitivity_allows(model: ModelConfig, sensitivity: str) -> bool:
        if sensitivity == "private":
            # Fail closed: only an explicitly no-train policy is suitable for private data.
            return model.data_policy == "no_train"
        return sensitivity == "public"
