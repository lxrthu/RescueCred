from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from environments.toolsandbox.adapter import canonical_action


METHODS = ("mask", "v4")
NONZERO_DECISIONS = ("rescue_preference", "reverse_preference")


def canonical_completion(action: Mapping[str, Any]) -> str:
    """Canonical action completion shared by training and evaluation."""

    return json.dumps(
        canonical_action(action),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def public_preference_prompt(
    *,
    visible_history: Sequence[Mapping[str, Any]],
    public_tool_schemas: Sequence[Mapping[str, Any]],
    action_a: Mapping[str, Any],
    action_b: Mapping[str, Any],
) -> str:
    """Build the model-visible candidate-selection prompt.

    The caller supplies only deployment-visible history and public schemas.  No
    evaluator output, trajectory suffix, milestone, minefield, or preference
    label is accepted by this interface.
    """

    payload = {
        "visible_history": list(visible_history),
        "public_tool_schemas": list(public_tool_schemas),
        "candidate_a": canonical_action(action_a),
        "candidate_b": canonical_action(action_b),
    }
    return (
        "Choose the better next tool action using only the visible history and "
        "public tool schemas. Return exactly one complete canonical JSON tool "
        "action, with no explanation.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\nSelected action:"
    )


def training_preference(
    row: Mapping[str, Any], method: str
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Return a same-budget Mask or causal-direction preference.

    Unit weights deliberately isolate the direction supplied by causal credit.
    Both methods consume the exact same event presentations.
    """

    if method not in METHODS:
        raise ValueError(f"unknown ToolSandbox preference method: {method}")
    if row.get("replay_valid") is not True:
        raise ValueError("preference training requires a replay-valid event")
    decision = str(row.get("decision"))
    if decision not in NONZERO_DECISIONS:
        raise ValueError("preference training requires nonzero causal credit")
    action_a = canonical_action(row["action_a"])
    action_b = canonical_action(row["action_b"])
    if method == "mask" or decision == "rescue_preference":
        return action_b, action_a, 1.0
    return action_a, action_b, 1.0


def matched_epoch_order(
    rows: Sequence[Mapping[str, Any]], seed: int, epoch: int
) -> list[Mapping[str, Any]]:
    """Deterministic identical event order for both compared methods."""

    ordered = list(rows)
    random.Random(seed + epoch * 1009 + 421337).shuffle(ordered)
    return ordered


def event_set_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    event_ids = sorted(str(row["event_id"]) for row in rows)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event ids must be unique")
    return hashlib.sha256("\n".join(event_ids).encode("utf-8")).hexdigest()


def summarize_evaluation_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if row.get("replay_valid") is True]
    decisions = Counter(str(row["decision"]) for row in valid)
    correct = [bool(row["causal_correct"]) for row in valid]
    rescue = [
        bool(row["causal_correct"])
        for row in valid
        if row["decision"] == "rescue_preference"
    ]
    reverse = [
        bool(row["causal_correct"])
        for row in valid
        if row["decision"] == "reverse_preference"
    ]

    def mean(values: Sequence[float]) -> float:
        return sum(float(value) for value in values) / max(1, len(values))

    return {
        "events": len(rows),
        "valid_events": len(valid),
        "decisions": dict(sorted(decisions.items())),
        "causal_accuracy": mean(correct),
        "rescue_accuracy": mean(rescue),
        "reverse_accuracy": mean(reverse),
        "selected_b_rate": mean([row["selected"] == "b" for row in valid]),
        "mean_selected_terminal_similarity": mean(
            [row["selected_terminal_similarity"] for row in valid]
        ),
        "mean_selected_progress_auc": mean(
            [row["selected_progress_auc"] for row in valid]
        ),
    }
