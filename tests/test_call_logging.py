import json

from llmrouter.call_logging import CallLogger


def test_jsonl_logger_redacts_configured_secrets_and_credential_patterns(tmp_path) -> None:
    path = tmp_path / "router.jsonl"
    logger = CallLogger(path, secrets=["provider-secret-value"])
    logger.emit(
        "attempt_finished", error="provider-secret-value Bearer token-value",
        detail="nvapi-1234567890abcdef",
    )
    raw = path.read_text(encoding="utf-8")
    record = json.loads(raw)
    assert "provider-secret-value" not in raw
    assert "token-value" not in raw
    assert "nvapi-1234567890abcdef" not in raw
    assert record["error"] == "[redacted] Bearer [redacted]"
    assert record["detail"] == "[redacted]"
