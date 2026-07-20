from __future__ import annotations

import ast
import hashlib
import hmac
import json
from typing import Any, Mapping, Sequence

from rescuecredit.deltaguard_observers import ObserverPredicate, validate_public_plan


def canonical_action(action: Mapping[str, Any]) -> dict[str, Any]:
    tool = action.get("tool")
    arguments = action.get("arguments")
    if not isinstance(tool, str) or not tool or not isinstance(arguments, Mapping):
        raise ValueError("action must contain a tool and mapping arguments")
    return {"tool": tool, "arguments": dict(arguments)}


def action_hash(action: Mapping[str, Any]) -> str:
    payload = json.dumps(
        canonical_action(action), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hmac_fraction(key: str, event_id: str) -> float:
    digest = hmac.new(key.encode("utf-8"), event_id.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest, "big") / float(1 << 256)


def acquisition_selected(
    *, event_id: str, eligible: bool, key: str, rate: float
) -> bool:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("acquisition rate must be in [0, 1]")
    return bool(eligible and hmac_fraction(key, event_id) < rate)


def parse_public_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    text = content.strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return text


def _value_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def public_context_digest(runtime: Any, context: Any) -> str:
    payload = {
        "visible_history": runtime.visible_history(context),
        "public_tool_schemas": runtime.tool_schemas(context),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _receipt_row(receipt: Any) -> dict[str, Any]:
    content = getattr(receipt, "content", None)
    exception = getattr(receipt, "exception", None)
    parsed = None if exception else parse_public_content(content)
    return {
        "action": canonical_action(getattr(receipt, "action")),
        "content": content,
        "exception": exception,
        "parsed": parsed,
        "value_hash": _value_hash(parsed if exception is None else {"exception": exception}),
    }


def execute_observer_plan(
    runtime: Any,
    context: Any,
    plan: Sequence[ObserverPredicate],
) -> tuple[list[dict[str, Any]], Any]:
    rows: list[dict[str, Any]] = []
    current = context
    for predicate in plan:
        receipt = runtime.execute(
            current,
            {"tool": predicate.tool, "arguments": predicate.arguments},
        )
        row = _receipt_row(receipt)
        row["predicate_id"] = predicate.predicate_id
        rows.append(row)
        current = receipt.context
    return rows, current


def run_paired_probe(
    *,
    runtime: Any,
    prefix: Any,
    action_a: Mapping[str, Any],
    action_b: Mapping[str, Any],
    plan: Sequence[ObserverPredicate],
) -> dict[str, Any]:
    """Execute identical public observations around isolated A/B branches."""

    action_a = canonical_action(action_a)
    action_b = canonical_action(action_b)
    if action_a == action_b:
        raise ValueError("paired probe requires distinct A/B actions")
    schemas = runtime.tool_schemas(prefix)
    validate_public_plan(plan, schemas)
    prefix_digest = public_context_digest(runtime, prefix)

    pre_rows, _ = execute_observer_plan(runtime, runtime.snapshot(prefix), plan)
    if public_context_digest(runtime, prefix) != prefix_digest:
        raise RuntimeError("pre-observation mutated the frozen prefix")

    receipt_a = runtime.execute(runtime.snapshot(prefix), action_a)
    post_a, _ = execute_observer_plan(runtime, receipt_a.context, plan)
    if public_context_digest(runtime, prefix) != prefix_digest:
        raise RuntimeError("A branch mutated the frozen prefix")

    receipt_b = runtime.execute(runtime.snapshot(prefix), action_b)
    post_b, _ = execute_observer_plan(runtime, receipt_b.context, plan)
    if public_context_digest(runtime, prefix) != prefix_digest:
        raise RuntimeError("B branch mutated the frozen prefix")

    return {
        "action_a": action_a,
        "action_b": action_b,
        "action_hash_a": action_hash(action_a),
        "action_hash_b": action_hash(action_b),
        "observer_plan": [predicate.to_dict() for predicate in plan],
        "pre_observations": pre_rows,
        "branch_a": {"action_receipt": _receipt_row(receipt_a), "post_observations": post_a},
        "branch_b": {"action_receipt": _receipt_row(receipt_b), "post_observations": post_b},
        "prefix_unchanged": public_context_digest(runtime, prefix) == prefix_digest,
        "official_evaluator_called": False,
        "hidden_state_exported": False,
    }
