"""Cloudflare Workers AI REST adapter using the account AI Run endpoint."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx

from llmrouter.adapters._http import class_for_status, raise_for_provider_error, redact_error
from llmrouter.adapters.base import ErrorClass, NormalizedRequest, NormalizedResponse, ProviderError
from llmrouter.config import ModelConfig


class CloudflareAdapter:
    name = "cloudflare"

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        self._async_transport = transport
        self._client = httpx.Client(transport=transport, trust_env=False)

    def complete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse:
        if not model.base_url:
            raise ProviderError(
                f"Model {model.id!r} needs a Cloudflare account API base_url",
                ErrorClass.CLIENT_ERROR,
            )
        api_token = os.environ.get(model.api_key_env)
        if not api_token:
            raise ProviderError(
                f"Required environment variable {model.api_key_env!r} is not set",
                ErrorClass.AUTH_ERROR,
            )
        body = _request_body(req, model.model_name)

        try:
            response = self._client.post(
                _run_url(model.base_url, model.model_name),
                headers={"authorization": f"Bearer {api_token}", "content-type": "application/json"},
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
            return _parse_response(payload, response.headers)
        except (ValueError, TypeError, KeyError) as exc:
            raise ProviderError(
                "Provider returned an invalid Workers AI response", ErrorClass.SERVER_ERROR,
                http_status=response.status_code,
            ) from exc

    async def acomplete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse:
        if not model.base_url:
            raise ProviderError(f"Model {model.id!r} needs a Cloudflare account API base_url", ErrorClass.CLIENT_ERROR)
        api_token = os.environ.get(model.api_key_env)
        if not api_token:
            raise ProviderError(f"Required environment variable {model.api_key_env!r} is not set", ErrorClass.AUTH_ERROR)
        body = _request_body(req, model.model_name)
        try:
            async with httpx.AsyncClient(
                transport=self._async_transport, trust_env=False
            ) as client:
                response = await client.post(
                    _run_url(model.base_url, model.model_name),
                    headers={"authorization": f"Bearer {api_token}", "content-type": "application/json"},
                    json=body, timeout=req.timeout_s,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError("Provider request timed out", ErrorClass.TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(redact_error(str(exc)), ErrorClass.SERVER_ERROR) from exc
        raise_for_provider_error(response)
        try:
            payload = response.json()
            return _parse_response(payload, response.headers)
        except (ValueError, TypeError, KeyError) as exc:
            raise ProviderError("Provider returned an invalid Workers AI response", ErrorClass.SERVER_ERROR, http_status=response.status_code) from exc

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


def _run_url(base_url: str, model_name: str) -> str:
    normalized = base_url.rstrip("/")
    # The unified AI model catalog uses /ai/run with a model field in the body.
    # Legacy Workers AI models use /ai/run/{model}.
    if not model_name.startswith("@"):
        return normalized if normalized.endswith("/ai/run") else f"{normalized}/ai/run"
    encoded_model = quote(model_name.strip("/"), safe="@/-._")
    if "/ai/run/" in normalized:
        return normalized
    if normalized.endswith("/ai/run"):
        return f"{normalized}/{encoded_model}"
    return f"{normalized}/ai/run/{encoded_model}"


def _request_body(req: NormalizedRequest, model_name: str) -> dict[str, Any]:
    if model_name.startswith("@"):
        body: dict[str, Any] = {"messages": [message.model_dump() for message in req.messages]}
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.max_output_tokens is not None:
            body["max_tokens"] = req.max_output_tokens
        return body

    contents: list[dict[str, Any]] = []
    system_parts: list[str] = []
    for message in req.messages:
        if message.role == "system":
            system_parts.append(message.content)
        elif message.role in {"user", "assistant", "model"}:
            role = "model" if message.role == "assistant" else message.role
            contents.append({"role": role, "parts": [{"text": message.content}]})
        else:
            raise ProviderError(
                f"Cloudflare Gemini adapter does not support message role {message.role!r}",
                ErrorClass.CLIENT_ERROR,
            )
    if not contents:
        raise ProviderError("Cloudflare Gemini requires at least one user or model message", ErrorClass.CLIENT_ERROR)

    request_input: dict[str, Any] = {"contents": contents}
    if system_parts:
        request_input["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    generation_config: dict[str, Any] = {}
    if req.temperature is not None:
        generation_config["temperature"] = req.temperature
    if req.max_output_tokens is not None:
        generation_config["maxOutputTokens"] = req.max_output_tokens
    if req.json_mode:
        generation_config["responseMimeType"] = "application/json"
    if generation_config:
        request_input["generationConfig"] = generation_config
    return {"model": model_name, "input": request_input}


def _parse_response(payload: Any, headers: Any) -> NormalizedResponse:
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        raise TypeError("result was not an object")

    text = result.get("response")
    usage = result.get("usage") or payload.get("usage") or {}
    finish_reason = result.get("finish_reason")
    if not isinstance(text, str):
        candidates = result.get("candidates") or []
        if not candidates:
            raise TypeError("result did not contain a text response")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        finish_reason = candidate.get("finishReason")
        usage = result.get("usageMetadata") or usage
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
    else:
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
    if not text:
        raise TypeError("result did not contain a text response")
    return NormalizedResponse(
        text=text,
        input_tokens=max(0, int(input_tokens or 0)),
        output_tokens=max(0, int(output_tokens or 0)),
        finish_reason=finish_reason,
        raw_headers={key.lower(): value for key, value in headers.items()},
        raw=payload,
    )
