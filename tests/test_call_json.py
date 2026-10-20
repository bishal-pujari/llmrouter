from pathlib import Path

import pytest
from pydantic import BaseModel

from llmrouter.adapters.base import NormalizedResponse
from llmrouter.adapters.fake import ScriptedFakeAdapter
from llmrouter.config import RouterConfig
from llmrouter.errors import OutputValidationError
from llmrouter.router import Router
from llmrouter.store.sqlite import SQLiteStore


class Answer(BaseModel):
    answer: str
    count: int


def make_config(second_model: bool = False, supports_json_mode: bool = True) -> RouterConfig:
    models = []
    for model_id, tier in ([('first', 'T0'), ('second', 'T1')] if second_model else [('first', 'T0')]):
        models.append({
            "id": model_id, "provider": "fake", "model_name": model_id,
            "api_key_env": "UNUSED", "tier": tier, "priority": 1, "enabled": True,
            "context_window": 8192, "max_output_tokens": 256,
            "input_price_per_mtok": 0, "output_price_per_mtok": 0,
            "is_free_tier": True, "data_policy": "no_train",
            "supports_json_mode": supports_json_mode,
        })
    tiers = ["T0", "T1"] if second_model else ["T0"]
    return RouterConfig.model_validate({
        "tiers": tiers,
        "models": models,
        "task_routes": {"extract": {
            "tier_order": tiers, "requires": {}, "default_temperature": 0,
            "max_attempts": 3, "budget_group": "default", "sensitivity": "public",
        }},
        "budgets": {"default": {"monthly_cap_usd": 10, "hard_stop": True}},
    })


def response(text: str) -> NormalizedResponse:
    return NormalizedResponse(text=text, input_tokens=12, output_tokens=8)


def make_router(tmp_path: Path, fake: ScriptedFakeAdapter, **config_options) -> Router:
    return Router(
        make_config(**config_options),
        SQLiteStore(tmp_path / "router.db"),
        adapters={"fake": fake},
    )


def test_call_json_parses_pydantic_and_uses_native_json_mode(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter({"first": [response('{"answer":"OK","count":1}')]})
    router = make_router(tmp_path, fake)
    try:
        result = router.call_json("extract", [{"role": "user", "content": "Extract it"}], Answer)
        assert result.data == Answer(answer="OK", count=1)
        assert fake.requests[0].json_mode is True
        assert fake.requests[0].response_format == {"type": "json_object"}
        assert "JSON Schema" in fake.requests[0].messages[0].content
    finally:
        router.close()


def test_call_json_retries_same_model_with_correction_and_strips_fence(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter({"first": [
        response('{"answer":"bad","count":"wrong"}'),
        response('```json\n{"answer":"OK","count":1}\n```'),
    ]})
    router = make_router(tmp_path, fake)
    try:
        result = router.call_json("extract", [{"role": "user", "content": "Extract it"}], Answer)
        assert result.data.answer == "OK"
        assert [attempt.status for attempt in result.attempts] == ["invalid_output", "ok"]
        assert [model_id for model_id, _ in fake.calls] == ["first", "first"]
        assert "Correct these issues" in fake.requests[1].messages[-1].content
        assert fake.requests[1].response_format == {"type": "json_object"}
    finally:
        router.close()


def test_call_json_escalates_after_two_failures_on_tier(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter({
        "first": [response("not json"), response("still not json")],
        "second": [response('{"answer":"OK","count":2}')],
    })
    router = make_router(tmp_path, fake, second_model=True)
    try:
        result = router.call_json("extract", [{"role": "user", "content": "Extract it"}], Answer)
        assert result.model_id == "second"
        assert [item.model_id for item in result.attempts] == ["first", "first", "second"]
        assert [item.status for item in result.attempts] == ["invalid_output", "invalid_output", "ok"]
        report = router.report()
        assert report["by_model"]["first"]["validation_failure_rate"] == pytest.approx(1.0)
        assert report["by_model"]["second"]["validation_failure_rate"] == pytest.approx(0.0)
    finally:
        router.close()


def test_call_json_supports_schema_validator_and_prompt_only_fallback(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter({"first": [response('{"answer":"no"}'), response('{"answer":"yes"}')]})
    router = make_router(tmp_path, fake, supports_json_mode=False)
    schema = {
        "type": "object", "properties": {"answer": {"type": "string"}},
        "required": ["answer"], "additionalProperties": False,
    }
    try:
        result = router.call_json(
            "extract", [{"role": "user", "content": "Extract it"}], schema,
            validator=lambda value: [] if value["answer"] == "yes" else ["answer must be yes"],
        )
        assert result.data == {"answer": "yes"}
        assert fake.requests[0].json_mode is False
        assert fake.requests[0].response_format is None
        assert "JSON Schema" in fake.requests[0].messages[0].content
        assert [item.status for item in result.attempts] == ["invalid_output", "ok"]
    finally:
        router.close()


def test_call_json_raises_typed_error_with_attempt_history(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter({"first": [response("no"), response("still no")]})
    router = make_router(tmp_path, fake)
    try:
        with pytest.raises(OutputValidationError) as captured:
            router.call_json("extract", [{"role": "user", "content": "Extract it"}], Answer)
        assert len(captured.value.attempts) == 2
        assert all(item.status == "invalid_output" for item in captured.value.attempts)
    finally:
        router.close()


def test_test_model_forces_one_configured_model_without_caching(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter({"first": [response("OK")]})
    router = make_router(tmp_path, fake)
    try:
        result = router.test_model("first")
        assert result.model_id == "first"
        assert result.text == "OK"
        assert fake.requests[0].max_output_tokens == 16
        assert router.store.cache_stats()["entries"] == 0
    finally:
        router.close()
