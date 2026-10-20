"""Strict YAML config models and loading helpers.

Extra fields are rejected so misspelled limits or prices cannot be silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llmrouter.errors import ConfigError


class StrictModel(BaseModel):
    # NaN/Infinity are not valid operational values for prices, limits, or
    # timeouts and can otherwise bypass ordinary ``ge``/``le`` checks.
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, allow_inf_nan=False
    )


class ModelConfig(StrictModel):
    id: str
    provider: str
    model_name: str
    base_url: str | None = None
    api_key_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    tier: str
    priority: int = 100
    enabled: bool = True
    rpm: int | None = Field(default=None, ge=1)
    rps: int | None = Field(default=None, ge=1)
    rpd: int | None = Field(default=None, ge=1)
    tpm: int | None = Field(default=None, ge=1)
    tpd: int | None = Field(default=None, ge=1)
    max_concurrency: int = Field(default=4, ge=1)
    context_window: int = Field(ge=0)
    max_output_tokens: int = Field(ge=0)
    supports_json_mode: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    input_price_per_mtok: float = Field(ge=0)
    output_price_per_mtok: float = Field(ge=0)
    shadow_input_price_per_mtok: float | None = Field(default=None, ge=0)
    shadow_output_price_per_mtok: float | None = Field(default=None, ge=0)
    is_free_tier: bool = False
    data_policy: str = "unknown"
    notes: str = ""

    @model_validator(mode="after")
    def validate_endpoint(self) -> "ModelConfig":
        """Keep credentials on TLS endpoints and reject URL-embedded secrets.

        Custom providers remain supported, but an accidental query string,
        userinfo component, or non-local cleartext endpoint must not receive an
        API key. Cleartext HTTP is limited to loopback for local test servers.
        """
        if self.base_url is None:
            return self
        try:
            parsed = urlsplit(self.base_url)
            hostname = parsed.hostname
            parsed.port  # Force malformed ports to fail validation.
        except ValueError as exc:
            raise ValueError("base_url is not a valid URL") from exc
        if parsed.scheme not in {"http", "https"} or not hostname:
            raise ValueError("base_url must be an http(s) URL with a hostname")
        if parsed.username or parsed.password:
            raise ValueError("base_url must not contain username or password")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query string or fragment")
        if parsed.scheme == "http" and hostname.lower() not in {
            "localhost", "127.0.0.1", "::1"
        }:
            raise ValueError("base_url must use HTTPS unless it targets loopback")
        return self


class TaskRoute(StrictModel):
    tier_order: list[str] = Field(min_length=1)
    requires: dict[str, Any] = Field(default_factory=dict)
    default_temperature: float = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    budget_group: str
    sensitivity: str = "public"
    cache_ttl_seconds: int | None = Field(default=None, ge=0)


class BudgetConfig(StrictModel):
    monthly_cap_usd: float = Field(ge=0)
    soft_warn_pct: float = Field(default=80, ge=0, le=100)
    hard_stop: bool = True


class Settings(StrictModel):
    default_timeout_s: float = Field(default=60, gt=0)
    limit_safety_fraction: float = Field(default=0.9, gt=0, le=1)
    log_payloads: bool = False
    cache_enabled: bool = True
    max_wait_s: float = Field(default=20, ge=0)


class RouterConfig(StrictModel):
    settings: Settings = Field(default_factory=Settings)
    tiers: list[str] = Field(min_length=1)
    models: list[ModelConfig]
    task_routes: dict[str, TaskRoute]
    budgets: dict[str, BudgetConfig]

    @model_validator(mode="after")
    def validate_references(self) -> "RouterConfig":
        model_ids = [model.id for model in self.models]
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("model IDs must be unique")
        tier_set = set(self.tiers)
        for model in self.models:
            if model.tier not in tier_set:
                raise ValueError(f"model {model.id!r} uses undefined tier {model.tier!r}")
        for task, route in self.task_routes.items():
            missing_tiers = set(route.tier_order) - tier_set
            if missing_tiers:
                raise ValueError(f"task {task!r} references undefined tiers: {sorted(missing_tiers)}")
            if route.budget_group not in self.budgets:
                raise ValueError(f"task {task!r} references undefined budget {route.budget_group!r}")
        return self


def load_config(path: str | Path) -> RouterConfig:
    """Load YAML and validate every field and cross-reference."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ConfigError(f"Configuration at {config_path} must be a YAML mapping")
        return RouterConfig.model_validate(raw)
    except OSError as exc:
        raise ConfigError(f"Cannot read configuration at {config_path}: {exc}") from exc
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ConfigError(f"Invalid configuration at {config_path}: {exc}") from exc
