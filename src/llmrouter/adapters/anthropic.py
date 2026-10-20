"""Anthropic Messages API adapter using plain httpx."""

from __future__ import annotations

import os
from typing import Any

import httpx

from llmrouter.adapters._http import class_for_status, raise_for_provider_error, redact_error
from llmrouter.adapters.base import ErrorClass, NormalizedRequest, NormalizedResponse, ProviderError
from llmrouter.config import ModelConfig


class AnthropicAdapter:
    name = "anthropic"

    def __init__(
        self, *, transport: httpx.BaseTransport | None = None,
        base_url: str = "https://api.anthropic.com",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._async_transport = transport
        self._client = httpx.Client(transport=transport, trust_env=False)

    def complete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse:
        api_key = os.environ.get(model.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Required environment variable {model.api_key_env!r} is not set",
                ErrorClass.AUTH_ERROR,
            )
        max_tokens = req.max_output_tokens or model.max_output_tokens
        if max_tokens < 1:
            raise ProviderError("Anthropic requests require max_output_tokens", ErrorClass.CLIENT_ERROR)

        system_parts: list[str] = []
        messages: list[dict[str, str]] = []
        for message in req.messages:
            if message.role == "system":
                system_parts.append(message.content)
            elif message.role in {"user", "assistant"}:
                messages.append({"role": message.role, "content": message.content})
            else:
                raise ProviderError(
                    f"Anthropic adapter does not support message role {message.role!r}",
                    ErrorClass.CLIENT_ERROR,
                )
        if not messages:
            raise ProviderError("Anthropic requires at least one user or assistant message", ErrorClass.CLIENT_ERROR)
        body: dict[str, Any] = {
            "model": model.model_name,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if req.temperature is not None:
            body["temperature"] = req.temperature

        try:
            response = self._client.post(
                _messages_url(model.base_url or self.base_url),
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
                timeout=req.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError("Provider request timed out", ErrorClass.TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(redact_error(str(exc)), ErrorClass.SERVER_ERROR) from exc
        raise_for_provider_error(response)

        try:
            payload = response.json()
            blocks = payload["content"]
            text = "".join(
                item.get("text", "") for item in blocks
                if isinstance(item, dict) and item.get("type") == "text"
            )
            usage = payload.get("usage") or {}
            if not isinstance(text, str) or not text:
                raise TypeError("message did not contain text")
            return NormalizedResponse(
                text=text,
                input_tokens=max(0, int(usage.get("input_tokens", 0) or 0)),
                output_tokens=max(0, int(usage.get("output_tokens", 0) or 0)),
                finish_reason=payload.get("stop_reason"),
                raw_headers={key.lower(): value for key, value in response.headers.items()},
                raw=payload,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise ProviderError(
                "Provider returned an invalid Messages response", ErrorClass.SERVER_ERROR,
                http_status=response.status_code,
            ) from exc

    async def acomplete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse:
        api_key = os.environ.get(model.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Required environment variable {model.api_key_env!r} is not set",
                ErrorClass.AUTH_ERROR,
            )
        max_tokens = req.max_output_tokens or model.max_output_tokens
        if max_tokens < 1:
            raise ProviderError("Anthropic requests require max_output_tokens", ErrorClass.CLIENT_ERROR)
        system_parts: list[str] = []
        messages: list[dict[str, str]] = []
        for message in req.messages:
            if message.role == "system":
                system_parts.append(message.content)
            elif message.role in {"user", "assistant"}:
                messages.append({"role": message.role, "content": message.content})
            else:
                raise ProviderError(f"Anthropic adapter does not support message role {message.role!r}", ErrorClass.CLIENT_ERROR)
        if not messages:
            raise ProviderError("Anthropic requires at least one user or assistant message", ErrorClass.CLIENT_ERROR)
        body: dict[str, Any] = {"model": model.model_name, "max_tokens": max_tokens, "messages": messages}
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if req.temperature is not None:
            body["temperature"] = req.temperature
        try:
            async with httpx.AsyncClient(
                transport=self._async_transport, trust_env=False
            ) as client:
                response = await client.post(
                    _messages_url(model.base_url or self.base_url),
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                    json=body, timeout=req.timeout_s,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError("Provider request timed out", ErrorClass.TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(redact_error(str(exc)), ErrorClass.SERVER_ERROR) from exc
        raise_for_provider_error(response)
        try:
            payload = response.json()
            blocks = payload["content"]
            text = "".join(item.get("text", "") for item in blocks if isinstance(item, dict) and item.get("type") == "text")
            usage = payload.get("usage") or {}
            if not text:
                raise TypeError("message did not contain text")
            return NormalizedResponse(
                text=text, input_tokens=max(0, int(usage.get("input_tokens", 0) or 0)),
                output_tokens=max(0, int(usage.get("output_tokens", 0) or 0)),
                finish_reason=payload.get("stop_reason"),
                raw_headers={key.lower(): value for key, value in response.headers.items()}, raw=payload,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise ProviderError("Provider returned an invalid Messages response", ErrorClass.SERVER_ERROR, http_status=response.status_code) from exc

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def classify_error(exc_or_response: object) -> ErrorClass:
        if isinstance(exc_or_response, ProviderError):
            return exc_or_response.error_class
        if isinstance(exc_or_response, httpx.TimeoutException):
            return ErrorClass.TIMEOUT
        if isinstance(exc_or_response, httpx.HTTPStatusError):
            return class_for_status(exc_or_response.response.status_code)
        return ErrorClass.SERVER_ERROR


def _messages_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/messages"):
        return normalized
    return f"{normalized}/messages" if normalized.endswith("/v1") else f"{normalized}/v1/messages"
