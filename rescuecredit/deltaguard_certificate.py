from __future__ import annotations

from typing import Any, Mapping, Sequence

from rescuecredit.deltaguard_goal_contract import goal_predicate_rows
from rescuecredit.deltaguard_observers import ObserverPredicate


UNKNOWN = "unknown"


def _receipt_quality(result: Mapping[str, Any]) -> int | str:
    """Score only explicit execution validity; never infer task correctness."""

    exception = result.get("exception")
    if not exception:
        return 1
    text = str(exception).casefold()
    if any(token in text for token in ("already", "unchanged", "no change")):
        return UNKNOWN
    return 0


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
    receipt_a = evidence.get("branch_a", {}).get("action_receipt", {})
    receipt_b = evidence.get("branch_b", {}).get("action_receipt", {})
    quality_a = _receipt_quality(receipt_a)
    quality_b = _receipt_quality(receipt_b)
    receipt_known = quality_a != UNKNOWN and quality_b != UNKNOWN
    receipt_required = not plan
    required_unknown = receipt_required and not receipt_known
    receipt_predicate_id = "receipt:explicit_execution_validity"
    if receipt_known:
        known_a.append(int(quality_a))
        known_b.append(int(quality_b))
        if int(quality_a) > int(quality_b):
            witness_a.append(receipt_predicate_id)
        elif int(quality_b) > int(quality_a):
            witness_b.append(receipt_predicate_id)
    rows.append(
        {
            "predicate_id": receipt_predicate_id,
            "family": str(evidence.get("receipt_family") or "execution"),
            "required": receipt_required,
            "pre": None,
            "post_a": quality_a,
            "post_b": quality_b,
            "delta_a": quality_a,
            "delta_b": quality_b,
            "pre_value_hash": None,
            "a_value_hash": receipt_a.get("value_hash"),
            "b_value_hash": receipt_b.get("value_hash"),
            "evidence_scope": "explicit action receipt exception only",
        }
    )
    contract = evidence.get("goal_contract")
    if isinstance(contract, Mapping):
        for goal_row in goal_predicate_rows(contract, evidence):
            delta_a = goal_row.get("delta_a", UNKNOWN)
            delta_b = goal_row.get("delta_b", UNKNOWN)
            known = delta_a != UNKNOWN and delta_b != UNKNOWN
            required_unknown = required_unknown or (
                bool(goal_row.get("required")) and not known
            )
            if known and bool(goal_row.get("routing_admissible")):
                known_a.append(int(delta_a))
                known_b.append(int(delta_b))
                if int(delta_a) > int(delta_b):
                    witness_a.append(str(goal_row["predicate_id"]))
                elif int(delta_b) > int(delta_a):
                    witness_b.append(str(goal_row["predicate_id"]))
            rows.append(
                {
                    "pre": None,
                    "post_a": delta_a,
                    "post_b": delta_b,
                    "pre_value_hash": None,
                    "a_value_hash": None,
                    "b_value_hash": None,
                    **goal_row,
                }
            )
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
        "version": (
            "public-paired-delta-v3-goal-contract"
            if isinstance(contract, Mapping)
            else "public-paired-delta-v2"
        ),
        "action_hash_a": evidence.get("action_hash_a"),
        "action_hash_b": evidence.get("action_hash_b"),
        "relation": relation,
        "reverse_score": score,
        "route_to_a": bool(score == 1.0),
        "required_unknown": required_unknown,
        "known_predicates": len(known_a),
        "total_predicates": len(rows),
        "witness_predicates": witness,
        "predicates": rows,
        "prefix_unchanged": evidence.get("prefix_unchanged") is True,
        "goal_contract_sha256": (
            contract.get("sha256") if isinstance(contract, Mapping) else None
        ),
    }
