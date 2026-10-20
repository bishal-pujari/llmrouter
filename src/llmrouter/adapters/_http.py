"""Small HTTP/error helpers shared by the focused REST adapters."""

from __future__ import annotations

import re
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from llmrouter.adapters.base import ErrorClass, ProviderError


def raise_for_provider_error(response: httpx.Response) -> None:
    if not response.is_error:
        return
    message = f"Provider returned HTTP {response.status_code}"
    # Never copy an untrusted provider error body into logs/exceptions: some
    # gateways echo submitted prompts or include request headers. The status
    # is enough for retry/fallback classification and remains actionable.
    raise ProviderError(
        message,
        class_for_status(response.status_code),
        http_status=response.status_code,
        retry_after_s=retry_after_seconds(response.headers.get("retry-after")),
    )


def class_for_status(status: int) -> ErrorClass:
    if status == 429:
        return ErrorClass.RATE_LIMITED
    if status in (401, 403):
        return ErrorClass.AUTH_ERROR
    if status in (408, 504):
        return ErrorClass.TIMEOUT
    if 500 <= status <= 599:
        return ErrorClass.SERVER_ERROR
    return ErrorClass.CLIENT_ERROR


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        delay = float(value)
        return max(0.0, delay) if math.isfinite(delay) else None
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def redact_error(message: str) -> str:
    return _redact(message)[:500]


def _redact(message: str) -> str:
    message = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", message)
    message = re.sub(r"\b(?:sk-|nvapi-)[A-Za-z0-9_-]+", "[redacted]", message)
    message = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)\S+", r"\1[redacted]", message)
    return message[:500]
