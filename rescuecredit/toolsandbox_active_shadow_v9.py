from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

from rescuecredit.toolsandbox_active_shadow import (
    ERROR_TERMS,
    PROTECTED_RECEIPT_KEYS,
    SUCCESS_TERMS,
    build_active_shadow_features,
)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:+-]+")


def _text(value: Any, limit: int = 4096) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)[:limit]


def _tokens(value: Any) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(_text(value))}


def _contains(text: str, terms: set[str]) -> float:
    lowered = text.lower()
    return float(any(term in lowered for term in terms))


def _protected_keys(value: Any) -> set[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return set()
    if isinstance(value, Mapping):
        found = {
            str(key)
            for key in value
            if str(key).casefold() in PROTECTED_RECEIPT_KEYS
        }
        for child in value.values():
            found.update(_protected_keys(child))
        return found
    if isinstance(value, (list, tuple)):
        found = set()
        for child in value:
            found.update(_protected_keys(child))
        return found
    return set()


def _argument_count(action: Any) -> int:
    if not isinstance(action, Mapping):
        return 0
    arguments = action.get("arguments")
    return len(arguments) if isinstance(arguments, Mapping) else 0


def _tool(action: Any) -> str:
    return str(action.get("tool", "")) if isinstance(action, Mapping) else ""


def _branch_features(branch: Mapping[str, Any]) -> tuple[list[float], list[Any]]:
    receipts = branch.get("receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("two-step ActiveShadow branch lacks a first receipt")
    first = receipts[0]
    if not isinstance(first, Mapping):
        raise ValueError("two-step ActiveShadow first receipt is malformed")
    second = receipts[1] if len(receipts) >= 2 else None
    if second is not None and not isinstance(second, Mapping):
        raise ValueError("two-step ActiveShadow second receipt is malformed")
    protected = set()
    for receipt in (first, second):
        if isinstance(receipt, Mapping):
            protected.update(_protected_keys(receipt.get("content")))
            protected.update(_protected_keys(receipt.get("exception")))
    if protected:
        raise ValueError(
            f"two-step ActiveShadow receipt contains protected keys: {sorted(protected)}"
        )

    first_content = _text(first.get("content"))
    first_exception = _text(first.get("exception"))
    second_content = _text(second.get("content")) if second else ""
    second_exception = _text(second.get("exception")) if second else ""
    first_action = first.get("action")
    second_action = second.get("action") if second else None
    receipt_tokens = _tokens(first_content)
    action_tokens = _tokens(second_action)
    grounding = len(receipt_tokens & action_tokens) / max(1, len(action_tokens))
    scalars = [
        float(second is not None),
        float(bool(second_exception)),
        _contains(second_content + " " + second_exception, ERROR_TERMS),
        _contains(second_content, SUCCESS_TERMS),
        math.log1p(len(second_content)) / 10.0,
        float(bool(second) and _tool(first_action) == _tool(second_action)),
        _argument_count(second_action) / 10.0,
        grounding,
        float(bool(first_exception) and bool(second) and not second_exception),
        float(not first_exception and bool(second_exception)),
        float(bool(second) and first_content == second_content),
    ]
    values = [first_action, first_content, first_exception]
    if second:
        values.extend([second_action, second_content, second_exception])
    return scalars, values


def _signed_hash(values: list[tuple[str, Any]], dimension: int) -> list[float]:
    result = [0.0] * dimension
    for namespace, value in values:
        for token in _tokens(value):
            digest = hashlib.sha256((namespace + ":" + token).encode()).digest()
            index = int.from_bytes(digest[:8], "big") % dimension
            result[index] += 1.0 if digest[8] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in result))
    return [value / norm for value in result] if norm else result


def build_two_step_features(
    row: Mapping[str, Any], *, hash_dimension: int
) -> list[float]:
    """Encode only A/B actions and their first two deployment-visible receipts."""

    branch_a = row.get("branch_a")
    branch_b = row.get("branch_b")
    if not isinstance(branch_a, Mapping) or not isinstance(branch_b, Mapping):
        raise ValueError("two-step ActiveShadow row lacks A/B branches")
    base = build_active_shadow_features(row, hash_dimension=hash_dimension)
    scalars_a, values_a = _branch_features(branch_a)
    scalars_b, values_b = _branch_features(branch_b)
    delta = [right - left for left, right in zip(scalars_a, scalars_b, strict=True)]
    hashed = _signed_hash(
        [("branch_a", values_a), ("branch_b", values_b)], hash_dimension
    )
    return base + scalars_a + scalars_b + delta + hashed
