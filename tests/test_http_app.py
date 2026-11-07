from __future__ import annotations

import json

import pytest

from llmrouter.http_app import RouterASGI
from llmrouter.router import CallResult


class FakeRouter:
    async def acall(self, task, messages, **options):
        return CallResult(
            text="OK", model_id="fake", cost_usd=0.0, cached=False, attempts=[],
            input_tokens=1, output_tokens=1, request_id="request-1",
        )

    async def acall_json(self, task, messages, schema, **options):
        return await self.acall(task, messages, **options)

    async def status(self):
        return [{"id": "fake", "enabled": True}]


async def request(app, method, path, payload=None, token="secret"):
    body = b"" if payload is None else json.dumps(payload).encode()
    events = [{"type": "http.request", "body": body, "more_body": False}]
    responses = []

    async def receive():
        return events.pop(0)

    async def send(message):
        responses.append(message)

    scope = {
        "type": "http", "method": method, "path": path,
        "headers": [(b"authorization", f"Bearer {token}".encode())],
    }
    await app(scope, receive, send)
    start = responses[0]
    return start["status"], json.loads(responses[1]["body"])


@pytest.mark.asyncio
async def test_http_call_requires_bearer_and_returns_stable_result_shape():
    app = RouterASGI(lambda _scope: FakeRouter(), auth_resolver=lambda _scope: "secret")
    status, payload = await request(app, "POST", "/v1/call", {
        "task": "summarize",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert status == 200
    assert payload["text"] == "OK"
    assert "attempts" not in payload

    unauthorized, body = await request(app, "POST", "/v1/call", {
        "task": "summarize",
        "messages": [{"role": "user", "content": "hello"}],
    }, token="wrong")
    assert unauthorized == 401
    assert body == {"error": "unauthorized"}


@pytest.mark.asyncio
async def test_health_is_public_but_invalid_payload_does_not_echo_prompt():
    app = RouterASGI(lambda _scope: FakeRouter(), auth_resolver=lambda _scope: "secret")
    status, payload = await request(app, "GET", "/healthz", token="wrong")
    assert status == 200
    assert payload == {"status": "ok"}

    secret_prompt = "do not echo this prompt " * 20_000
    status, payload = await request(app, "POST", "/v1/call", {
        "task": "summarize",
        "messages": [{"role": "user", "content": secret_prompt}],
        "unknown_field": True,
    })
    assert status == 400
    assert payload == {"error": "invalid_request"}
