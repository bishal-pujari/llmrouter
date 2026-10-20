"""OpenAI-compatible chat completion adapter implemented with plain httpx."""

from __future__ import annotations

import os
import re
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from llmrouter.adapters.base import (
    ErrorClass,
    NormalizedRequest,
    NormalizedResponse,
    ProviderError,
)
from llmrouter.config import ModelConfig


class OpenAICompatAdapter:
    name = "openai_compat"

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        # A transport hook makes adapter and router tests fully offline.
        self._async_transport = transport
        # Do not let ambient HTTP(S)_PROXY/SSL environment variables redirect
        # credential-bearing requests outside the configured endpoint.
        self._client = httpx.Client(transport=transport, trust_env=False)

    def complete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse:
        if not model.base_url:
            raise ProviderError(
                f"Model {model.id!r} has no base_url", ErrorClass.CLIENT_ERROR
            )
        api_key = os.environ.get(model.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Required environment variable {model.api_key_env!r} is not set",
                ErrorClass.AUTH_ERROR,
            )

        url = _completion_url(model.base_url)
        body: dict[str, Any] = {
            "model": model.model_name,
            "messages": [message.model_dump() for message in req.messages],
        }
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.max_output_tokens is not None:
            body["max_tokens"] = req.max_output_tokens
        if req.response_format is not None:
            body["response_format"] = req.response_format

        try:
            response = self._client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=req.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError("Provider request timed out", ErrorClass.TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(_redact(str(exc)), ErrorClass.SERVER_ERROR) from exc

        if response.is_error:
            raise _http_error(response)
        try:
            payload = response.json()
            choice = payload["choices"][0]
            text = choice["message"]["content"]
            if not isinstance(text, str):
                raise TypeError("assistant content was not text")
            usage = payload.get("usage") or {}
            return NormalizedResponse(
                text=text,
                input_tokens=max(0, int(usage.get("prompt_tokens", 0) or 0)),
                output_tokens=max(0, int(usage.get("completion_tokens", 0) or 0)),
                finish_reason=choice.get("finish_reason"),
                raw_headers={key.lower(): value for key, value in response.headers.items()},
                raw=payload,
            )
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise ProviderError(
                "Provider returned an invalid chat completion response", ErrorClass.SERVER_ERROR,
                http_status=response.status_code,
            ) from exc

    async def acomplete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse:
        if not model.base_url:
            raise ProviderError(f"Model {model.id!r} has no base_url", ErrorClass.CLIENT_ERROR)
        api_key = os.environ.get(model.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Required environment variable {model.api_key_env!r} is not set",
                ErrorClass.AUTH_ERROR,
            )
        body: dict[str, Any] = {
            "model": model.model_name,
            "messages": [message.model_dump() for message in req.messages],
        }
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.max_output_tokens is not None:
            body["max_tokens"] = req.max_output_tokens
        if req.response_format is not None:
            body["response_format"] = req.response_format
        try:
            async with httpx.AsyncClient(
                transport=self._async_transport, trust_env=False
            ) as client:
                response = await client.post(
                    _completion_url(model.base_url),
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=body,
                    timeout=req.timeout_s,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError("Provider request timed out", ErrorClass.TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(_redact(str(exc)), ErrorClass.SERVER_ERROR) from exc
        if response.is_error:
            raise _http_error(response)
        try:
            payload = response.json()
            choice = payload["choices"][0]
            text = choice["message"]["content"]
            if not isinstance(text, str):
                raise TypeError("assistant content was not text")
            usage = payload.get("usage") or {}
            return NormalizedResponse(
                text=text,
                input_tokens=max(0, int(usage.get("prompt_tokens", 0) or 0)),
                output_tokens=max(0, int(usage.get("completion_tokens", 0) or 0)),
                finish_reason=choice.get("finish_reason"),
                raw_headers={key.lower(): value for key, value in response.headers.items()},
                raw=payload,
            )
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise ProviderError(
                "Provider returned an invalid chat completion response", ErrorClass.SERVER_ERROR,
                http_status=response.status_code,
            ) from exc

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def classify_error(exc_or_response: object) -> ErrorClass:
        if isinstance(exc_or_response, ProviderError):
            return exc_or_response.error_class
        if isinstance(exc_or_response, httpx.TimeoutException):
            return ErrorClass.TIMEOUT
        if isinstance(exc_or_response, httpx.HTTPStatusError):
            status = exc_or_response.response.status_code
            return _class_for_status(status)
        if isinstance(exc_or_response, httpx.HTTPError):
            return ErrorClass.SERVER_ERROR
        return ErrorClass.SERVER_ERROR


def _completion_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _class_for_status(status: int) -> ErrorClass:
    if status == 429:
        return ErrorClass.RATE_LIMITED
    if status in (401, 403):
        return ErrorClass.AUTH_ERROR
    if status in (408, 504):
        return ErrorClass.TIMEOUT
    if 500 <= status <= 599:
        return ErrorClass.SERVER_ERROR
    if status == 400:
        return ErrorClass.CLIENT_ERROR
    return ErrorClass.CLIENT_ERROR


def _http_error(response: httpx.Response) -> ProviderError:
    error_message = f"Provider returned HTTP {response.status_code}"
    return ProviderError(
        error_message,
        _class_for_status(response.status_code),
        http_status=response.status_code,
        retry_after_s=_retry_after_seconds(response.headers.get("retry-after")),
    )


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        delay = float(value)
        return max(0.0, delay) if math.isfinite(delay) else None
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _redact(message: str) -> str:
    message = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", message)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", message)
    message = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)\S+", r"\1[redacted]", message)
    return message[:500]
