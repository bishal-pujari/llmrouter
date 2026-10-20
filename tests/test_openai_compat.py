from pathlib import Path
import json

import httpx
import pytest

from llmrouter.adapters.openai_compat import OpenAICompatAdapter
from llmrouter.adapters.base import NormalizedRequest
from llmrouter.config import RouterConfig
from llmrouter.router import Router
from llmrouter.store.sqlite import SQLiteStore


def make_config(second_model: bool = False) -> RouterConfig:
    models = [
        {
            "id": "first", "provider": "openai_compat", "model_name": "example-model-a",
            "base_url": "https://provider.example/v1", "api_key_env": "TEST_LLM_KEY",
            "tier": "T0", "priority": 1, "enabled": True, "context_window": 4096,
            "max_output_tokens": 64, "input_price_per_mtok": 1.0,
            "output_price_per_mtok": 1.0, "is_free_tier": False, "data_policy": "no_train",
        }
    ]
    tier_order = ["T0"]
    tiers = ["T0"]
    if second_model:
        models.append({
            **models[0], "id": "second", "model_name": "example-model-b", "tier": "T1",
            "priority": 1,
        })
        tiers = ["T0", "T1"]
        tier_order = tiers
    return RouterConfig.model_validate({
        "tiers": tiers,
        "models": models,
        "task_routes": {
            "smoke": {
                "tier_order": tier_order, "requires": {}, "default_temperature": 0,
                "max_attempts": 3, "budget_group": "default", "sensitivity": "public",
            }
        },
        "budgets": {"default": {"monthly_cap_usd": 10, "hard_stop": True}},
    })


def test_openai_compatible_adapter_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "unit-test-secret")
    seen = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["authorization"]
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        })

    adapter = OpenAICompatAdapter(transport=httpx.MockTransport(handle))
    config = make_config()
    try:
        response = adapter.complete(
            NormalizedRequest(
                messages=[{"role": "user", "content": "hi"}], request_id="req"
            ),
            config.models[0],
        )
    finally:
        adapter.close()
    assert response.text == "hello"
    assert response.input_tokens == 4
    assert response.output_tokens == 2
    assert seen["url"] == "https://provider.example/v1/chat/completions"
    assert seen["authorization"] == "Bearer unit-test-secret"
    assert "example-model-a" in seen["body"]


@pytest.mark.asyncio
async def test_openai_compatible_adapter_native_async_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "unit-test-secret")
    seen = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "async hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })

    adapter = OpenAICompatAdapter(transport=httpx.MockTransport(handle))
    try:
        response = await adapter.acomplete(
            NormalizedRequest(messages=[{"role": "user", "content": "hi"}], request_id="async-req"),
            make_config().models[0],
        )
    finally:
        adapter.close()
    assert response.text == "async hello"
    assert (response.input_tokens, response.output_tokens) == (3, 2)
    assert seen["url"] == "https://provider.example/v1/chat/completions"


def test_router_falls_back_after_429_and_updates_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "unit-test-secret")
    config = make_config(second_model=True)

    def handle(request: httpx.Request) -> httpx.Response:
        model_name = request.read().decode()
        if "example-model-a" in model_name:
            return httpx.Response(
                429, headers={"Retry-After": "30"}, json={"error": {"message": "slow down"}}
            )
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "fallback worked"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        })

    adapter = OpenAICompatAdapter(transport=httpx.MockTransport(handle))
    router = Router(config, SQLiteStore(tmp_path / "router.db"), adapters={"openai_compat": adapter})
    try:
        result = router.call("smoke", [{"role": "user", "content": "hello"}])
        assert result.text == "fallback worked"
        assert result.model_id == "second"
        assert [attempt.status for attempt in result.attempts] == ["rate_limited", "ok"]
        status = {item["model_id"]: item for item in router.status()}
        assert status["first"]["cooldown"]["reason"] == "rate limited"
        assert status["first"]["cooldown"]["until"] > 0
        assert status["second"]["usage"]["minute"]["requests"] == 1
        rows = router.store.connection.execute(
            "SELECT status, http_status, prompt_hash FROM call_log ORDER BY attempt_no"
        ).fetchall()
        assert [(row["status"], row["http_status"]) for row in rows] == [
            ("rate_limited", 429), ("ok", None)
        ]
        assert all(len(row["prompt_hash"]) == 64 for row in rows)
        log_records = [
            json.loads(line)
            for line in (tmp_path / "router.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [record["event"] for record in log_records].count("api_call_started") == 2
        assert [record["event"] for record in log_records].count("api_call_finished") == 2
        assert all("hello" not in line for line in (tmp_path / "router.jsonl").read_text().splitlines())
    finally:
        router.close()
