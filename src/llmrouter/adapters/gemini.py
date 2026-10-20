"""Google Gemini generateContent REST adapter implemented with httpx."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx

from llmrouter.adapters._http import class_for_status, raise_for_provider_error, redact_error
from llmrouter.adapters.base import ErrorClass, NormalizedRequest, NormalizedResponse, ProviderError
from llmrouter.config import ModelConfig


class GeminiAdapter:
    name = "gemini"

    def __init__(
        self, *, transport: httpx.BaseTransport | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
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
                    f"Gemini adapter does not support message role {message.role!r}",
                    ErrorClass.CLIENT_ERROR,
                )
        if not contents:
            raise ProviderError("Gemini requires at least one user or model message", ErrorClass.CLIENT_ERROR)
        body: dict[str, Any] = {"contents": contents}
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        generation_config: dict[str, Any] = {}
        if req.temperature is not None:
            generation_config["temperature"] = req.temperature
        if req.max_output_tokens is not None:
            generation_config["maxOutputTokens"] = req.max_output_tokens
        if req.json_mode:
            generation_config["responseMimeType"] = "application/json"
        if generation_config:
            body["generationConfig"] = generation_config

        try:
            response = self._client.post(
                _generate_url(model.base_url or self.base_url, model.model_name),
                headers={"x-goog-api-key": api_key, "content-type": "application/json"},
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
            candidates = payload.get("candidates") or []
            if not candidates:
                feedback = payload.get("promptFeedback") or {}
                if feedback.get("blockReason"):
                    raise ProviderError("Gemini blocked the prompt", ErrorClass.BLOCKED)
                raise ValueError("no response candidates")
            candidate = candidates[0]
            finish_reason = candidate.get("finishReason")
            if finish_reason in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY"}:
                raise ProviderError("Gemini blocked the response", ErrorClass.BLOCKED)
            parts = (candidate.get("content") or {}).get("parts") or []
            text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
            usage = payload.get("usageMetadata") or {}
            if not text:
                raise ValueError("response contained no text")
            return NormalizedResponse(
                text=text,
                input_tokens=max(0, int(usage.get("promptTokenCount", 0) or 0)),
                output_tokens=max(0, int(usage.get("candidatesTokenCount", 0) or 0)),
                finish_reason=finish_reason,
                raw_headers={key.lower(): value for key, value in response.headers.items()},
                raw=payload,
            )
        except ProviderError:
            raise
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise ProviderError(
                "Provider returned an invalid generateContent response", ErrorClass.SERVER_ERROR,
                http_status=response.status_code,
            ) from exc

    async def acomplete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse:
        api_key = os.environ.get(model.api_key_env)
        if not api_key:
            raise ProviderError(f"Required environment variable {model.api_key_env!r} is not set", ErrorClass.AUTH_ERROR)
        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []
        for message in req.messages:
            if message.role == "system":
                system_parts.append(message.content)
            elif message.role in {"user", "assistant", "model"}:
                role = "model" if message.role == "assistant" else message.role
                contents.append({"role": role, "parts": [{"text": message.content}]})
            else:
                raise ProviderError(f"Gemini adapter does not support message role {message.role!r}", ErrorClass.CLIENT_ERROR)
        if not contents:
            raise ProviderError("Gemini requires at least one user or model message", ErrorClass.CLIENT_ERROR)
        body: dict[str, Any] = {"contents": contents}
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        generation_config: dict[str, Any] = {}
        if req.temperature is not None:
            generation_config["temperature"] = req.temperature
        if req.max_output_tokens is not None:
            generation_config["maxOutputTokens"] = req.max_output_tokens
        if req.json_mode:
            generation_config["responseMimeType"] = "application/json"
        if generation_config:
            body["generationConfig"] = generation_config
        try:
            async with httpx.AsyncClient(
                transport=self._async_transport, trust_env=False
            ) as client:
                response = await client.post(
                    _generate_url(model.base_url or self.base_url, model.model_name),
                    headers={"x-goog-api-key": api_key, "content-type": "application/json"},
                    json=body, timeout=req.timeout_s,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError("Provider request timed out", ErrorClass.TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(redact_error(str(exc)), ErrorClass.SERVER_ERROR) from exc
        raise_for_provider_error(response)
        try:
            payload = response.json()
            candidates = payload.get("candidates") or []
            if not candidates:
                feedback = payload.get("promptFeedback") or {}
                if feedback.get("blockReason"):
                    raise ProviderError("Gemini blocked the prompt", ErrorClass.BLOCKED)
                raise ValueError("no response candidates")
            candidate = candidates[0]
            finish_reason = candidate.get("finishReason")
            if finish_reason in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY"}:
                raise ProviderError("Gemini blocked the response", ErrorClass.BLOCKED)
            parts = (candidate.get("content") or {}).get("parts") or []
            text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
            usage = payload.get("usageMetadata") or {}
            if not text:
                raise ValueError("response contained no text")
            return NormalizedResponse(
                text=text, input_tokens=max(0, int(usage.get("promptTokenCount", 0) or 0)),
                output_tokens=max(0, int(usage.get("candidatesTokenCount", 0) or 0)),
                finish_reason=finish_reason,
                raw_headers={key.lower(): value for key, value in response.headers.items()}, raw=payload,
            )
        except ProviderError:
            raise
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise ProviderError("Provider returned an invalid generateContent response", ErrorClass.SERVER_ERROR, http_status=response.status_code) from exc

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


def _generate_url(base_url: str, model_name: str) -> str:
    normalized = base_url.rstrip("/")
    if "/models/" in normalized and normalized.endswith(":generateContent"):
        return normalized
    model = model_name.removeprefix("models/").strip("/")
    encoded_model = quote(model, safe="-._")
    return f"{normalized}/models/{encoded_model}:generateContent"
