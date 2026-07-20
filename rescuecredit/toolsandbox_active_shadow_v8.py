from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

from rescuecredit.toolsandbox_active_shadow import build_active_shadow_features


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:+-]+")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value[:8192]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)[:8192]


def _hash(values: list[tuple[str, Any]], dimension: int) -> list[float]:
    result = [0.0] * dimension
    for namespace, value in values:
        for token in TOKEN_PATTERN.findall(_text(value).lower()):
            digest = hashlib.sha256((namespace + ":" + token).encode()).digest()
            index = int.from_bytes(digest[:8], "big") % dimension
            result[index] += 1.0 if digest[8] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in result))
    return [value / norm for value in result] if norm else result


def _schema_names(schemas: Any) -> set[str]:
    names = set()
    if not isinstance(schemas, list):
        return names
    for schema in schemas:
        if not isinstance(schema, Mapping):
            continue
        function = schema.get("function", schema)
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            names.add(str(function["name"]))
    return names


def _summary_scalars(summary: Mapping[str, Any]) -> list[float]:
    before = _schema_names(summary.get("schemas_before"))
    after = _schema_names(summary.get("schemas_after"))
    appended = summary.get("appended_visible_history", [])
    if not isinstance(appended, list):
        raise ValueError("visible-state summary has malformed history")
    exception_rows = sum(
        bool(row.get("tool_call_exception"))
        for row in appended
        if isinstance(row, Mapping)
    )
    return [
        len(appended) / 10.0,
        exception_rows / max(1.0, len(appended)),
        len(after - before) / 10.0,
        len(before - after) / 10.0,
        float(before == after),
    ]


def build_v8_features(row: Mapping[str, Any], *, hash_dimension: int) -> list[float]:
    summary_a = row.get("state_summary_a")
    summary_b = row.get("state_summary_b")
    if not isinstance(summary_a, Mapping) or not isinstance(summary_b, Mapping):
        raise ValueError("V8 row lacks A/B visible-state summaries")
    pseudo = {
        "action_a": row.get("action_a"),
        "action_b": row.get("action_b"),
        "branch_a": {"receipts": [summary_a.get("receipt")]},
        "branch_b": {"receipts": [summary_b.get("receipt")]},
    }
    base = build_active_shadow_features(pseudo, hash_dimension=hash_dimension)
    scalars_a = _summary_scalars(summary_a)
    scalars_b = _summary_scalars(summary_b)
    delta = [right - left for left, right in zip(scalars_a, scalars_b, strict=True)]
    hashed = _hash(
        [
            ("history_a", summary_a.get("appended_visible_history")),
            ("history_b", summary_b.get("appended_visible_history")),
            ("schema_before", summary_a.get("schemas_before")),
            ("schema_after_a", summary_a.get("schemas_after")),
            ("schema_after_b", summary_b.get("schemas_after")),
        ],
        hash_dimension,
    )
    return base + scalars_a + scalars_b + delta + hashed
