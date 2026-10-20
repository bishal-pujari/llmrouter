import json

import httpx
import pytest

from llmrouter.adapters.anthropic import AnthropicAdapter
from llmrouter.adapters.base import ErrorClass, Message, NormalizedRequest, ProviderError
from llmrouter.adapters.cloudflare import CloudflareAdapter
from llmrouter.adapters.gemini import GeminiAdapter
from llmrouter.config import ModelConfig


def model(provider: str, *, base_url: str | None = None) -> ModelConfig:
    return ModelConfig.model_validate({
        "id": provider, "provider": provider, "model_name": "test-model",
        "base_url": base_url, "api_key_env": "PROVIDER_TEST_KEY", "tier": "T0",
        "context_window": 4096, "max_output_tokens": 64,
        "input_price_per_mtok": 0, "output_price_per_mtok": 0,
        "is_free_tier": True, "supports_json_mode": True,
    })


def request() -> NormalizedRequest:
    return NormalizedRequest(
        messages=[
            Message(role="system", content="Be concise."),
            Message(role="user", content="Question?"),
            Message(role="assistant", content="Prior answer."),
        ],
        temperature=0.2, max_output_tokens=24, request_id="request-1",
        json_mode=True, response_format={"type": "json_object"},
    )


def test_anthropic_messages_mapping_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_TEST_KEY", "anthropic-secret")
    seen: dict[str, object] = {}

    def handle(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["headers"] = dict(req.headers)
        seen["body"] = json.loads(req.read())
        return httpx.Response(200, json={
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "there"},
            ],
            "usage": {"input_tokens": 11, "output_tokens": 4},
            "stop_reason": "end_turn",
        })

    adapter = AnthropicAdapter(transport=httpx.MockTransport(handle))
    try:
        result = adapter.complete(request(), model("anthropic"))
    finally:
        adapter.close()
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "anthropic-secret"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    body = seen["body"]
    assert body["system"] == "Be concise."
    assert [message["role"] for message in body["messages"]] == ["user", "assistant"]
    assert body["max_tokens"] == 24
    assert result.text == "Hello there"
    assert (result.input_tokens, result.output_tokens, result.finish_reason) == (11, 4, "end_turn")


def test_gemini_request_mapping_native_json_mode_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_TEST_KEY", "gemini-secret")
    seen: dict[str, object] = {}

    def handle(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["headers"] = dict(req.headers)
        seen["body"] = json.loads(req.read())
        return httpx.Response(200, json={
            "candidates": [{
                "content": {"parts": [{"text": "{\"answer\":"}, {"text": "\"yes\"}"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 3},
        })

    adapter = GeminiAdapter(
        base_url="https://gemini.example/v1beta", transport=httpx.MockTransport(handle)
    )
    try:
        result = adapter.complete(request(), model("gemini", base_url="https://gemini.example/v1beta"))
    finally:
        adapter.close()
    assert seen["url"] == "https://gemini.example/v1beta/models/test-model:generateContent"
    assert seen["headers"]["x-goog-api-key"] == "gemini-secret"
    body = seen["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "Be concise."
    assert [message["role"] for message in body["contents"]] == ["user", "model"]
    assert body["generationConfig"]["maxOutputTokens"] == 24
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert result.text == '{"answer":"yes"}'
    assert (result.input_tokens, result.output_tokens, result.finish_reason) == (9, 3, "STOP")


def test_cloudflare_workers_ai_run_request_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_TEST_KEY", "cloudflare-secret")
    seen: dict[str, object] = {}

    def handle(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["headers"] = dict(req.headers)
        seen["body"] = json.loads(req.read())
        return httpx.Response(200, json={
            "success": True,
            "result": {
                "response": "Cloudflare reply",
                "usage": {"prompt_tokens": 7, "completion_tokens": 5},
            },
        })

    configured = model(
        "cloudflare", base_url="https://api.cloudflare.example/client/v4/accounts/account-123"
    ).model_copy(update={"model_name": "@cf/meta/test-model"})
    adapter = CloudflareAdapter(transport=httpx.MockTransport(handle))
    try:
        result = adapter.complete(request(), configured)
    finally:
        adapter.close()
    assert seen["url"] == (
        "https://api.cloudflare.example/client/v4/accounts/account-123/ai/run/@cf/meta/test-model"
    )
    assert seen["headers"]["authorization"] == "Bearer cloudflare-secret"
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["max_tokens"] == 24
    assert result.text == "Cloudflare reply"
    assert (result.input_tokens, result.output_tokens) == (7, 5)


def test_cloudflare_ai_catalog_gemini_request_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_TEST_KEY", "cloudflare-secret")
    seen: dict[str, object] = {}

    def handle(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.read())
        return httpx.Response(200, json={
            "result": {
                "candidates": [{
                    "content": {"parts": [{"text": "Cloudflare Gemini reply"}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 5},
            },
        })

    configured = model(
        "cloudflare", base_url="https://api.cloudflare.example/client/v4/accounts/account-123"
    ).model_copy(update={"model_name": "google/gemini-3.8-flash"})
    adapter = CloudflareAdapter(transport=httpx.MockTransport(handle))
    try:
        result = adapter.complete(request(), configured)
    finally:
        adapter.close()

    assert seen["url"] == "https://api.cloudflare.example/client/v4/accounts/account-123/ai/run"
    body = seen["body"]
    assert body["model"] == "google/gemini-3.8-flash"
    assert body["input"]["contents"] == [
        {"role": "user", "parts": [{"text": "Question?"}]},
        {"role": "model", "parts": [{"text": "Prior answer."}]},
    ]
    assert body["input"]["systemInstruction"]["parts"][0]["text"] == "Be concise."
    assert body["input"]["generationConfig"]["maxOutputTokens"] == 24
    assert result.text == "Cloudflare Gemini reply"
    assert (result.input_tokens, result.output_tokens, result.finish_reason) == (7, 5, "STOP")


@pytest.mark.parametrize(
    ("adapter_type", "provider", "status", "error_class"),
    [
        (AnthropicAdapter, "anthropic", 429, ErrorClass.RATE_LIMITED),
        (GeminiAdapter, "gemini", 401, ErrorClass.AUTH_ERROR),
        (CloudflareAdapter, "cloudflare", 503, ErrorClass.SERVER_ERROR),
    ],
)
def test_provider_http_errors_are_classified(
    monkeypatch: pytest.MonkeyPatch, adapter_type, provider, status, error_class,
) -> None:
    monkeypatch.setenv("PROVIDER_TEST_KEY", "test-key")

    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, headers={"Retry-After": "17"},
            json={"error": {"message": "upstream error"}},
        )

    adapter = adapter_type(transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(ProviderError) as captured:
            adapter.complete(request(), model(provider, base_url=(
                "https://api.cloudflare.example/client/v4/accounts/a" if provider == "cloudflare" else None
            )))
    finally:
        adapter.close()
    assert captured.value.error_class == error_class
    assert captured.value.http_status == status
    if status == 429:
        assert captured.value.retry_after_s == 17
