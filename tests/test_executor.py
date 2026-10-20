import random

from llmrouter.adapters.base import ErrorClass, NormalizedRequest, NormalizedResponse, ProviderError
from llmrouter.adapters.fake import ScriptedFakeAdapter
from llmrouter.config import ModelConfig
from llmrouter.executor import AllAttemptsFailed, execute


def model(model_id: str, provider: str = "fake") -> ModelConfig:
    return ModelConfig(
        id=model_id, provider=provider, model_name=model_id, api_key_env="UNUSED", tier="T0",
        context_window=1000, max_output_tokens=100, input_price_per_mtok=0,
        output_price_per_mtok=0,
    )


def request() -> NormalizedRequest:
    return NormalizedRequest(messages=[{"role": "user", "content": "hello"}], request_id="req-1")


def test_rate_limit_falls_back_and_preserves_request_id() -> None:
    adapter = ScriptedFakeAdapter({
        "first": [ProviderError("slow down", ErrorClass.RATE_LIMITED, http_status=429)],
        "second": [NormalizedResponse(text="ok")],
    })
    result = execute(request(), [model("first"), model("second")], {"fake": adapter}, max_attempts=3)
    assert result.model_id == "second"
    assert [item.status for item in result.attempts] == ["rate_limited", "ok"]
    assert [request_id for _, request_id in adapter.calls] == ["req-1", "req-1"]


def test_server_error_retries_same_model_once_then_falls_back() -> None:
    adapter = ScriptedFakeAdapter({
        "first": [
            ProviderError("temporary", ErrorClass.SERVER_ERROR, http_status=500),
            ProviderError("still down", ErrorClass.SERVER_ERROR, http_status=500),
        ],
        "second": [NormalizedResponse(text="recovered")],
    })
    result = execute(
        request(), [model("first"), model("second")], {"fake": adapter}, max_attempts=3,
        sleep=lambda _: None,
    )
    assert result.model_id == "second"
    assert [model_id for model_id, _ in adapter.calls] == ["first", "first", "second"]


def test_attempt_limit_returns_typed_history() -> None:
    adapter = ScriptedFakeAdapter({
        "only": [ProviderError("bad request", ErrorClass.CLIENT_ERROR, http_status=400)]
    })
    try:
        execute(request(), [model("only")], {"fake": adapter}, max_attempts=1)
    except AllAttemptsFailed as exc:
        assert len(exc.attempts) == 1
        assert exc.attempts[0].status == "client_error"
    else:
        raise AssertionError("expected all attempts to fail")


def test_seeded_failure_injection_always_returns_or_raises_typed_history() -> None:
    failures = [ErrorClass.RATE_LIMITED, ErrorClass.TIMEOUT, ErrorClass.SERVER_ERROR]
    candidates = [model(f"model-{index}") for index in range(4)]
    for seed in range(40):
        rng = random.Random(seed)
        scripts = {}
        for candidate in candidates:
            attempts = []
            for _ in range(2):
                if rng.random() < 0.35:
                    attempts.append(NormalizedResponse(text="valid result"))
                else:
                    failure = rng.choice(failures)
                    attempts.append(ProviderError(f"seed {seed} simulated", failure))
            scripts[candidate.id] = attempts
        adapter = ScriptedFakeAdapter(scripts)
        try:
            result = execute(
                request(), candidates, {"fake": adapter}, max_attempts=8,
                sleep=lambda _: None, random_value=lambda: 0.5,
            )
        except AllAttemptsFailed as exc:
            assert 0 < len(exc.attempts) <= 8
            assert all(attempt.status in {item.value for item in failures} for attempt in exc.attempts)
        else:
            assert result.response.text == "valid result"
            assert 1 <= len(result.attempts) <= 8
