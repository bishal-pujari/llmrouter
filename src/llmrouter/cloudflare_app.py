"""Cloudflare-specific ASGI application wiring."""

from __future__ import annotations

import asyncio
from typing import Any

from llmrouter.async_router import AsyncRouter
from llmrouter.http_app import RouterASGI


def env_value(env: Any, name: str) -> str | None:
    if isinstance(env, dict):
        value = env.get(name)
    else:
        value = getattr(env, name, None)
    if value is None:
        return None
    return str(value)


class CloudflareRouterProvider:
    def __init__(self) -> None:
        self._router: AsyncRouter | None = None
        self._lock = asyncio.Lock()

    async def __call__(self, scope: dict[str, Any]) -> AsyncRouter:
        env = scope.get("env")
        if env is None:
            raise RuntimeError("Cloudflare environment bindings are unavailable")
        if self._router is None:
            async with self._lock:
                if self._router is None:
                    config = env_value(env, "LLMROUTER_CONFIG_JSON")
                    database = getattr(env, "DB", None) if not isinstance(env, dict) else env.get("DB")
                    if not config or database is None:
                        raise RuntimeError("LLMROUTER_CONFIG_JSON and D1 binding DB are required")
                    self._router = await AsyncRouter.from_config_text(
                        config, database, format="json",
                        secret_resolver=lambda name: env_value(env, name),
                    )
        return self._router


_provider = CloudflareRouterProvider()
app = RouterASGI(_provider, auth_resolver=lambda scope: env_value(scope.get("env"), "LLMROUTER_HTTP_API_KEY"))
