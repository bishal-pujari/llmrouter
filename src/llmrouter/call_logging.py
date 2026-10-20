"""Prompt-free JSON Lines logs for provider attempt starts and outcomes."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any


_locks_guard = threading.Lock()
_path_locks: dict[str, threading.Lock] = {}


def default_log_path(db_path: str) -> Path:
    """Put the activity log beside the SQLite file, or under data for memory DBs."""
    if db_path == ":memory:":
        return Path("data/router.jsonl")
    return Path(db_path).with_name("router.jsonl")


class CallLogger:
    def __init__(self, path: str | Path, secrets: list[str] | None = None) -> None:
        self.path = Path(path)
        self._secrets = tuple(secret for secret in (secrets or []) if secret)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Logs can contain caller labels and provider error text. Keep the
        # file private even when the process umask is permissive, and refuse
        # to follow a pre-existing symlink at this trust boundary.
        with _open_private_append(self.path):
            pass
        lock_key = str(self.path.resolve())
        with _locks_guard:
            self._lock = _path_locks.setdefault(lock_key, threading.Lock())

    def emit(self, event: str, **fields: Any) -> None:
        """Append one compact JSON event; callers pass metadata only, never prompts."""
        record = {"ts": int(time.time()), "event": event, **_redact_fields(fields, self._secrets)}
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        with self._lock, _open_private_append(self.path) as stream:
            stream.write(line + "\n")


def _open_private_append(path: Path):
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    os.chmod(path, 0o600)
    return os.fdopen(descriptor, "a", encoding="utf-8")


def _redact_fields(fields: dict[str, Any], secrets: tuple[str, ...]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in fields.items():
        lowered_key = key.lower()
        if (
            any(marker in lowered_key for marker in ("api_key", "secret", "authorization"))
            or lowered_key in {"token", "access_token", "refresh_token"}
            or lowered_key.endswith("_token")
        ):
            redacted[key] = "[redacted]"
        else:
            redacted[key] = _redact_value(value, secrets)
    return redacted


def _redact_value(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return _redact_fields(value, secrets)
    if isinstance(value, list):
        return [_redact_value(item, secrets) for item in value]
    if isinstance(value, str):
        return _redact_text(value, secrets)
    return value


def _redact_text(value: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        value = value.replace(secret, "[redacted]")
    value = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[redacted]", value)
    value = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+", r"\1[redacted]", value)
    value = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{8,}|nvapi-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{20,})\b", "[redacted]", value)
    return value
