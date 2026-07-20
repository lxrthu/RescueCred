from __future__ import annotations

from typing import Any, Mapping, Sequence

from rescuecredit.deltaguard_observers import ObserverPredicate


UNKNOWN = "unknown"


def _matches(row: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    return all(row.get(key) == value for key, value in target.items())


def extract_value(result: Mapping[str, Any], predicate: ObserverPredicate) -> Any:
    if result.get("exception"):
        return UNKNOWN
    value = result.get("parsed")
    if predicate.extractor == "boolean":
        return value if isinstance(value, bool) else UNKNOWN
    if predicate.extractor == "count":
        return len(value) if isinstance(value, list) else UNKNOWN
    if predicate.extractor == "match_count":
        if not isinstance(value, list) or not isinstance(predicate.target, Mapping):
            return UNKNOWN
        rows = [row for row in value if isinstance(row, Mapping)]
        return sum(_matches(row, predicate.target) for row in rows)
    raise ValueError(f"unknown observer extractor: {predicate.extractor}")


def typed_delta(pre: Any, post: Any, predicate: ObserverPredicate) -> int | str:
    if pre == UNKNOWN or post == UNKNOWN:
        return UNKNOWN
    if predicate.comparator == "target":
        before = pre == predicate.target
        after = post == predicate.target
        return int(after) - int(before)
    if not isinstance(pre, (int, float)) or not isinstance(post, (int, float)):
        return UNKNOWN
    if predicate.comparator == "increase":
        return 1 if post > pre else -1 if post < pre else 0
    if predicate.comparator == "unique_add":
        if post == pre:
            return 0
        return 1 if pre == 0 and post == 1 else -1
    if predicate.comparator == "decrease":
        return 1 if post < pre else -1 if post > pre else 0
    raise ValueError(f"unknown observer comparator: {predicate.comparator}")


def _by_predicate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row.get("predicate_id")): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("observer results contain duplicate predicate identifiers")
    return result


def build_delta_certificate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    plan = [ObserverPredicate.from_dict(row) for row in evidence.get("observer_plan", [])]
    if not plan:
        raise ValueError("cannot certify an empty observer plan")
    pre = _by_predicate(evidence.get("pre_observations", []))
    post_a = _by_predicate(evidence.get("branch_a", {}).get("post_observations", []))
    post_b = _by_predicate(evidence.get("branch_b", {}).get("post_observations", []))
    expected = {predicate.predicate_id for predicate in plan}
    if set(pre) != expected or set(post_a) != expected or set(post_b) != expected:
        raise ValueError("observer evidence does not match the frozen plan")

    rows = []
    required_unknown = False
    known_a: list[int] = []
    known_b: list[int] = []
    witness_a: list[str] = []
    witness_b: list[str] = []
    for predicate in plan:
        pre_value = extract_value(pre[predicate.predicate_id], predicate)
        value_a = extract_value(post_a[predicate.predicate_id], predicate)
        value_b = extract_value(post_b[predicate.predicate_id], predicate)
        delta_a = typed_delta(pre_value, value_a, predicate)
        delta_b = typed_delta(pre_value, value_b, predicate)
        known = delta_a != UNKNOWN and delta_b != UNKNOWN
        required_unknown = required_unknown or (predicate.required and not known)
        if known:
            known_a.append(int(delta_a))
            known_b.append(int(delta_b))
            if int(delta_a) > int(delta_b):
                witness_a.append(predicate.predicate_id)
            elif int(delta_b) > int(delta_a):
                witness_b.append(predicate.predicate_id)
        rows.append(
            {
                "predicate_id": predicate.predicate_id,
                "family": predicate.family,
                "required": predicate.required,
                "pre": pre_value,
                "post_a": value_a,
                "post_b": value_b,
                "delta_a": delta_a,
                "delta_b": delta_b,
                "pre_value_hash": pre[predicate.predicate_id].get("value_hash"),
                "a_value_hash": post_a[predicate.predicate_id].get("value_hash"),
                "b_value_hash": post_b[predicate.predicate_id].get("value_hash"),
            }
        )

    if len(known_a) != len(known_b):
        raise ValueError("paired public delta length mismatch")
    paired_deltas = list(zip(known_a, known_b))
    a_dominates = bool(
        known_a
        and not required_unknown
        and all(left >= right for left, right in paired_deltas)
        and any(left > right for left, right in paired_deltas)
    )
    b_dominates = bool(
        known_a
        and not required_unknown
        and all(right >= left for left, right in paired_deltas)
        and any(right > left for left, right in paired_deltas)
    )
    if a_dominates:
        relation, score, witness = "a_dominates_b", 1.0, witness_a
    elif b_dominates:
        relation, score, witness = "b_dominates_a", 0.0, witness_b
    else:
        relation, score, witness = "incomparable", 0.5, []
    return {
        "version": "public-paired-delta-v1",
        "action_hash_a": evidence.get("action_hash_a"),
        "action_hash_b": evidence.get("action_hash_b"),
        "relation": relation,
        "reverse_score": score,
        "route_to_a": bool(score == 1.0),
        "required_unknown": required_unknown,
        "known_predicates": len(known_a),
        "total_predicates": len(plan),
        "witness_predicates": witness,
        "predicates": rows,
        "prefix_unchanged": evidence.get("prefix_unchanged") is True,
    }
