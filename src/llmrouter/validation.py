"""JSON output parsing and schema/custom validation helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from pydantic import BaseModel
from jsonschema import Draft202012Validator

from llmrouter.errors import OutputValidationError


JSONSchema = type[BaseModel] | dict[str, Any]
CustomValidator = Callable[[Any], list[str]]


def schema_as_json_schema(schema: JSONSchema) -> dict[str, Any]:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    if isinstance(schema, dict):
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:
            raise ValueError(f"Invalid JSON Schema: {exc}") from exc
        return schema
    raise TypeError("schema must be a Pydantic BaseModel class or a JSON Schema dictionary")


def json_instruction(schema: JSONSchema, supplied_system: str | None = None) -> str:
    schema_dict = schema_as_json_schema(schema)
    instructions = (
        "Return only one valid JSON value that conforms to this JSON Schema. "
        "Do not include Markdown fences or prose.\nJSON Schema:\n"
        + json.dumps(schema_dict, ensure_ascii=False, separators=(",", ":"))
    )
    return f"{supplied_system.rstrip()}\n\n{instructions}" if supplied_system else instructions


def validate_json_output(
    text: str,
    schema: JSONSchema,
    custom_validator: CustomValidator | None = None,
) -> Any:
    """Strip an optional code fence, parse JSON, then validate its structure."""
    candidate = _strip_code_fence(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(
            f"Output is not valid JSON: {exc.msg}",
            [f"JSON parse error at line {exc.lineno}, column {exc.colno}: {exc.msg}"],
        ) from exc

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            validated: Any = schema.model_validate(parsed)
        except Exception as exc:
            problems = _pydantic_problems(exc)
            raise OutputValidationError("JSON does not match the Pydantic model", problems) from exc
    else:
        schema_dict = schema_as_json_schema(schema)
        errors = sorted(Draft202012Validator(schema_dict).iter_errors(parsed), key=lambda err: list(err.path))
        if errors:
            problems = [
                f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
                for error in errors
            ]
            raise OutputValidationError("JSON does not match the JSON Schema", problems)
        validated = parsed

    if custom_validator is not None:
        problems = custom_validator(validated)
        if problems:
            raise OutputValidationError("Custom output validation failed", list(problems))
    return validated


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else stripped


def _pydantic_problems(exc: Exception) -> list[str]:
    if hasattr(exc, "errors"):
        try:
            return [
                f"{'/'.join(str(part) for part in error.get('loc', ())) or '<root>'}: "
                f"{error.get('msg', 'validation error')}"
                for error in exc.errors()
            ]
        except Exception:
            pass
    return [str(exc)[:300]]
