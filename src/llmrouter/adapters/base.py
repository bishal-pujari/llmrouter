"""Provider-neutral request/response types and adapter contract."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from llmrouter.config import ModelConfig


class ErrorClass(StrEnum):
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    CLIENT_ERROR = "client_error"
    AUTH_ERROR = "auth_error"
    BLOCKED = "blocked"


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str


class NormalizedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[Message] = Field(min_length=1)
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    request_id: str
    timeout_s: float = Field(default=60, gt=0)
    json_mode: bool = False
    response_format: dict[str, Any] | None = None


class NormalizedResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    finish_reason: str | None = None
    raw_headers: dict[str, str] = Field(default_factory=dict)
    raw: object | None = None


class ProviderError(Exception):
    """Classified provider failure safe for executor decisions."""

    def __init__(
        self,
        message: str,
        error_class: ErrorClass,
        *,
        http_status: int | None = None,
        retry_after_s: float | None = None,
        used_tokens: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.http_status = http_status
        self.retry_after_s = retry_after_s
        self.used_tokens = used_tokens


class ProviderAdapter(Protocol):
    name: str

    def complete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse: ...

    async def acomplete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse: ...

    def classify_error(self, exc_or_response: object) -> ErrorClass: ...
