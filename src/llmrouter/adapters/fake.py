"""Scriptable offline adapter for executor development and deterministic tests."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from llmrouter.adapters.base import ErrorClass, NormalizedRequest, NormalizedResponse, ProviderError
from llmrouter.config import ModelConfig


class ScriptedFakeAdapter:
    name = "fake"

    def __init__(self, scripts: dict[str, Iterable[NormalizedResponse | Exception]] | None = None) -> None:
        self._scripts = defaultdict(deque)
        for model_id, sequence in (scripts or {}).items():
            self._scripts[model_id].extend(sequence)
        self.calls: list[tuple[str, str]] = []
        self.requests: list[NormalizedRequest] = []

    def complete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse:
        self.calls.append((model.id, req.request_id))
        self.requests.append(req)
        if not self._scripts[model.id]:
            raise RuntimeError(f"No fake response scripted for model {model.id!r}")
        result = self._scripts[model.id].popleft()
        if isinstance(result, Exception):
            raise result
        return result

    async def acomplete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse:
        return self.complete(req, model)

    @staticmethod
    def classify_error(exc_or_response: object) -> ErrorClass:
        if isinstance(exc_or_response, ProviderError):
            return exc_or_response.error_class
        return ErrorClass.SERVER_ERROR
