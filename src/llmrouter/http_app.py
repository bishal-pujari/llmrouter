"""Small authenticated ASGI API for local and Worker deployments.

The wrapper intentionally has no framework dependency. It can run behind
Uvicorn locally and can be passed to Cloudflare's ``workers.asgi`` bridge.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llmrouter.errors import BudgetExceeded, ConfigError, NoModelAvailable
from llmrouter.executor import AllAttemptsFailed


class RouterLike(Protocol):
    async def acall(self, task: str, messages: list[dict[str, str]], **options: Any) -> Any: ...
    async def acall_json(self, task: str, messages: list[dict[str, str]], schema: object, **options: Any) -> Any: ...
    async def status(self) -> list[dict[str, object]]: ...


class MessageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str = Field(min_length=1, max_length=32)
    content: str = Field(min_length=1, max_length=100_000)


class CallInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")
    messages: list[MessageInput] = Field(min_length=1, max_length=64)
    caller: str | None = Field(default=None, max_length=200)
    system: str | None = Field(default=None, max_length=100_000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    sensitivity: str | None = Field(default=None, pattern=r"^(public|private)$")
    timeout_s: float | None = Field(default=None, gt=0, le=300)
    force_model: str | None = Field(default=None, max_length=200)
    use_cache: bool = True
    cache_ttl_seconds: int | None = Field(default=None, ge=0, le=31_536_000)
    max_cost_usd: float | None = Field(default=None, ge=0, le=1_000_000)


class JsonCallInput(CallInput):
    json_schema: dict[str, Any] = Field(alias="schema", min_length=1, max_length=100)


RouterProvider = Callable[[dict[str, Any]], RouterLike | Awaitable[RouterLike]]
SecretResolver = Callable[[dict[str, Any]], str | None]


class RequestTooLarge(ValueError):
    """The bounded HTTP body limit was exceeded."""


class RouterASGI:
    """ASGI application exposing only health, status, and call endpoints."""

    def __init__(self, router_provider: RouterProvider, *, auth_resolver: SecretResolver,
                 max_body_bytes: int = 512_000) -> None:
        self.router_provider = router_provider
        self.auth_resolver = auth_resolver
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict[str, Any], receive: Callable[[], Awaitable[dict[str, Any]]],
                       send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        if scope.get("type") != "http":
            await self._send(send, 404, {"error": "not_found"})
            return
        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", ""))
        if method == "GET" and path == "/healthz":
            await self._send(send, 200, {"status": "ok"})
            return
        if not self._authorized(scope):
            await self._send(send, 401, {"error": "unauthorized"}, headers={"www-authenticate": "Bearer"})
            return
        try:
            router = self.router_provider(scope)
            if asyncio.iscoroutine(router):
                router = await router
            if method == "GET" and path == "/v1/status":
                await self._send(send, 200, {"models": await _maybe_await(router.status())})
                return
            if method == "POST" and path in {"/v1/call", "/v1/call/json"}:
                payload = await self._body(receive)
                if path.endswith("/json"):
                    request = JsonCallInput.model_validate(payload)
                else:
                    request = CallInput.model_validate(payload)
                options = request.model_dump(exclude_none=True, exclude={"task", "messages", "json_schema"})
                messages = [item.model_dump() for item in request.messages]
                result = (
                    await _maybe_await(router.acall_json(request.task, messages, request.json_schema, **options))
                    if isinstance(request, JsonCallInput)
                    else await _maybe_await(router.acall(request.task, messages, **options))
                )
                await self._send(send, 200, _result_payload(result))
                return
            await self._send(send, 404, {"error": "not_found"})
        except RequestTooLarge:
            await self._send(send, 413, {"error": "request_too_large"})
        except (ValidationError, json.JSONDecodeError, ValueError, ConfigError):
            # Do not return Pydantic input details: they may contain sensitive prompts.
            await self._send(send, 400, {"error": "invalid_request"})
        except BudgetExceeded:
            await self._send(send, 429, {"error": "budget_exceeded"})
        except NoModelAvailable:
            await self._send(send, 503, {"error": "no_model_available"})
        except AllAttemptsFailed:
            await self._send(send, 502, {"error": "provider_unavailable"})
        except Exception:
            await self._send(send, 500, {"error": "internal_error"})

    def _authorized(self, scope: dict[str, Any]) -> bool:
        expected = self.auth_resolver(scope)
        if not expected:
            return False
        headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
        authorization = headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        return scheme.lower() == "bearer" and bool(token) and hmac.compare_digest(token, expected)

    async def _body(self, receive: Callable[[], Awaitable[dict[str, Any]]]) -> Any:
        chunks: list[bytes] = []
        total = 0
        while True:
            event = await receive()
            chunk = event.get("body", b"")
            total += len(chunk)
            if total > self.max_body_bytes:
                raise RequestTooLarge("request body too large")
            chunks.append(chunk)
            if not event.get("more_body", False):
                break
        return json.loads(b"".join(chunks))

    @staticmethod
    async def _send(send: Callable[[dict[str, Any]], Awaitable[None]], status: int,
                    payload: dict[str, object], headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        response_headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
        for key, value in (headers or {}).items():
            response_headers.append((key.lower().encode(), value.encode()))
        await send({"type": "http.response.start", "status": status, "headers": response_headers})
        await send({"type": "http.response.body", "body": body})


def _result_payload(result: Any) -> dict[str, object]:
    return {
        "text": result.text,
        "model_id": result.model_id,
        "cost_usd": result.cost_usd,
        "cached": result.cached,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "request_id": result.request_id,
        "data": result.data,
    }


async def _maybe_await(value: Any) -> Any:
    return await value if hasattr(value, "__await__") else value


def create_local_app(router: Any) -> RouterASGI:
    """Create an ASGI app around an existing local ``Router`` instance."""
    async def provider(_scope: dict[str, Any]) -> Any:
        return router

    return RouterASGI(provider, auth_resolver=lambda _scope: os.environ.get("LLMROUTER_HTTP_API_KEY"))
