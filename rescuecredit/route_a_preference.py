from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from typing import Any


def stable_rank(seed: int, event_id: str) -> str:
    return hashlib.sha256(f"{seed}:{event_id}".encode()).hexdigest()


def stratified_split(
    rows: list[dict[str, Any]], seed: int, validation_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["decision"])].append(row)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: stable_rank(seed, row["event_id"]))
        count = max(1, int(math.ceil(len(ordered) * validation_fraction)))
        count = min(count, max(1, len(ordered) - 1)) if len(ordered) > 1 else 1
        validation.extend(ordered[:count])
        train.extend(ordered[count:])
    return (
        sorted(train, key=lambda row: row["event_id"]),
        sorted(validation, key=lambda row: row["event_id"]),
    )


def training_preference(
    row: dict[str, Any], method: str, max_weight: float = 2.5
) -> tuple[dict[str, Any], dict[str, Any], float] | None:
    if method == "mask":
        return row["action_b"], row["action_a"], 1.0
    if method == "v31":
        relation = validity_relation(row)
        if relation == "a_invalid_b_valid":
            # Public schema validity is lexicographically prior to trajectory
            # credit. A missing-required-argument proposal is never taught as
            # preferable merely because a stochastic continuation recovered.
            return row["action_b"], row["action_a"], 1.0
        if relation == "a_valid_b_invalid":
            return row["action_a"], row["action_b"], 1.0
        if relation != "both_valid":
            return None
    elif method not in {"v2", "v3"}:
        raise ValueError(f"unknown method: {method}")
    delta = float(row["delta"])
    if abs(delta) <= 1e-12:
        return None
    weight = min(abs(delta), float(max_weight))
    if delta > 0:
        return row["action_b"], row["action_a"], weight
    return row["action_a"], row["action_b"], weight


def validity_relation(row: dict[str, Any]) -> str:
    """Infer only public, action-time executable validity for Route-A pairs."""

    a_valid = row.get("action_a_executable")
    b_valid = row.get("action_b_executable")
    if isinstance(a_valid, bool) and isinstance(b_valid, bool):
        if a_valid and not b_valid:
            return "a_valid_b_invalid"
        if b_valid and not a_valid:
            return "a_invalid_b_valid"
        return "both_valid" if a_valid else "both_invalid"
    if a_valid is not None or b_valid is not None:
        # Do not combine incomplete explicit metadata with a weaker fallback.
        return "unknown"
    kind = str(row.get("variant_kind", ""))
    if kind == "missing_required_arguments" or row.get("missing_parameter"):
        return "a_invalid_b_valid"
    if kind == "wrong_visible_candidate_value":
        return "both_valid"
    return "unknown"


def preference_kind(row: dict[str, Any], method: str) -> str:
    if method != "v31":
        return str(row.get("decision", "unknown"))
    relation = validity_relation(row)
    if relation == "a_invalid_b_valid":
        return "validity_b_over_a"
    if relation == "a_valid_b_invalid":
        return "validity_a_over_b"
    if relation == "both_valid":
        return str(row.get("decision", "unknown"))
    return "skipped_unknown_validity"


def completion(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def epoch_order(rows: list[dict[str, Any]], seed: int, epoch: int) -> list[dict[str, Any]]:
    ordered = list(rows)
    random.Random(seed + epoch * 1009).shuffle(ordered)
    return ordered
