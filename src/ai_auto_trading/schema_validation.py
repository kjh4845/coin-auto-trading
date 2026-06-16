from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
from pathlib import Path
from typing import Any


class SchemaValidationError(ValueError):
    """Raised when a payload does not satisfy the local JSON schema subset."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_trade_log_schema() -> dict[str, Any]:
    schema_path = _repo_root() / "docs" / "schemas" / "trade_log_v1.schema.json"
    return json.loads(schema_path.read_text())


def validate_trade_record(payload: Mapping[str, Any]) -> None:
    schema = load_trade_log_schema()
    _validate(payload, schema, "$")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return _is_number(value)
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    return False


def _validate(value: Any, schema: Mapping[str, Any], path: str) -> None:
    if "type" in schema:
        allowed = schema["type"]
        allowed_types = allowed if isinstance(allowed, list) else [allowed]
        if not any(_matches_type(value, item) for item in allowed_types):
            raise SchemaValidationError(f"{path}: expected type {allowed_types}, got {type(value).__name__}")

    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(f"{path}: expected const {schema['const']!r}, got {value!r}")

    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path}: expected one of {schema['enum']!r}, got {value!r}")

    if isinstance(value, str) and "minLength" in schema and len(value) < schema["minLength"]:
        raise SchemaValidationError(f"{path}: string shorter than minLength {schema['minLength']}")

    if _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path}: number smaller than minimum {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise SchemaValidationError(
                f"{path}: number must be > {schema['exclusiveMinimum']}"
            )

    if isinstance(value, str) and schema.get("format") == "date-time":
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaValidationError(f"{path}: invalid date-time {value!r}") from exc

    if isinstance(value, Mapping):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise SchemaValidationError(f"{path}: missing required key {key!r}")

        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise SchemaValidationError(f"{path}: unexpected keys {sorted(extra)!r}")

        for key, child in value.items():
            if key in properties:
                _validate(child, properties[key], f"{path}.{key}")

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{index}]")

