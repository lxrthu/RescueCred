#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_credit import (
    lexicographic_counterfactual_regret,
    validate_branch_credit_evidence,
)
from scripts.freeze_toolsandbox_v44_candidate_protocol import PROTOCOL_STATUS


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _task_bootstrap(rows: list[dict], replicates: int, seed: int) -> dict[str, Any]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id_hash"])].append(row)
    tasks = sorted(by_task)
    if not tasks:
        return {
            "replicates": float(replicates),
            "probability_rescue_observed": 0.0,
            "probability_reverse_observed": 0.0,
            "probability_both_directions_observed": 0.0,
            "rescue_prevalence": {"lower95": None, "median": None, "upper95": None},
            "reverse_prevalence": {"lower95": None, "median": None, "upper95": None},
            "oracle_headroom": {"lower95": None, "median": None, "upper95": None},
        }
    rng = random.Random(seed)
    rescue_seen = reverse_seen = both_seen = 0
    rescue_prevalence = []
    reverse_prevalence = []
    headroom = []
    for _ in range(replicates):
        sampled = []
        for _ in tasks:
            sampled.extend(by_task[rng.choice(tasks)])
        decisions = Counter(str(row["decision"]) for row in sampled)
        total = sum(decisions.values())
        rescue = decisions["rescue_preference"] / max(total, 1)
        reverse = decisions["reverse_preference"] / max(total, 1)
        has_rescue = decisions["rescue_preference"] > 0
        has_reverse = decisions["reverse_preference"] > 0
        rescue_seen += has_rescue
        reverse_seen += has_reverse
        both_seen += has_rescue and has_reverse
        rescue_prevalence.append(rescue)
        reverse_prevalence.append(reverse)
        headroom.append(min(rescue, reverse))

    def interval(values: list[float]) -> dict[str, float | None]:
        return {
            "lower95": _quantile(values, 0.025),
            "median": _quantile(values, 0.5),
            "upper95": _quantile(values, 0.975),
        }

    return {
        "replicates": float(replicates),
        "probability_rescue_observed": rescue_seen / replicates,
        "probability_reverse_observed": reverse_seen / replicates,
        "probability_both_directions_observed": both_seen / replicates,
        "rescue_prevalence": interval(rescue_prevalence),
        "reverse_prevalence": interval(reverse_prevalence),
        "oracle_headroom": interval(headroom),
    }


def _credit_equal(recomputed: dict[str, Any], row: dict[str, Any]) -> bool:
    return (
        recomputed["decision"] == row.get("decision")
        and recomputed["decision_basis"] == row.get("decision_basis")
        and math.isclose(
            float(recomputed["decision_value"]),
            float(row.get("decision_value")),
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
        and recomputed["components"] == row.get("credit_components")
        and row.get("credit_mode") == "lexicographic_v4"
        and math.isclose(
            float(recomputed["causal_weight"]),
            float(row.get("causal_weight")),
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--raw-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    audit = json.loads(args.audit_summary.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("fresh confirmation protocol is not frozen")
    rows = read_jsonl(args.raw_events)
    event_ids = [str(row.get("event_id", "")) for row in rows]
    if any(not event_id for event_id in event_ids) or len(event_ids) != len(set(event_ids)):
        raise ValueError("fresh confirmation event IDs are missing or duplicated")
    sealed_hashes = protocol.get("scenario_identity", {}).get("fresh_hashes", [])
    scenario_counts = Counter(str(row.get("scenario_name", "")) for row in rows)
    replay_identity_valid = all(
        bool(row.get("replay_valid"))
        == (
            row.get("branch_a", {}).get("valid") is True
            and row.get("branch_b", {}).get("valid") is True
        )
        for row in rows
    )
    event_derivation_valid = all(
        str(row.get("event_id", ""))
        == hashlib.sha256(
            (
                str(row.get("mode", ""))
                + "\0"
                + str(row.get("scenario_name", ""))
                + "\0prefix="
                + str(row.get("reference_free_prefix_steps", ""))
                + ":candidate="
                + str(row.get("candidate_rank", ""))
            ).encode("utf-8")
        ).hexdigest()
        for row in rows
    )
    identity_valid = all(
        hashlib.sha256(str(row.get("scenario_name", "")).encode("utf-8")).hexdigest()
        == str(row.get("task_id_hash", ""))
        and str(row.get("task_id_hash", "")) in set(sealed_hashes)
        for row in rows
    )
    credit_recomputed = True
    evidence_valid = True
    for row in rows:
        if row.get("replay_valid") is not True:
            continue
        try:
            validate_branch_credit_evidence(
                row["branch_a"], horizon=int(protocol["horizon"]), atol=float(protocol["atol"])
            )
            validate_branch_credit_evidence(
                row["branch_b"], horizon=int(protocol["horizon"]), atol=float(protocol["atol"])
            )
            recomputed = lexicographic_counterfactual_regret(
                row["branch_a"],
                row["branch_b"],
                horizon=int(protocol["horizon"]),
                atol=float(protocol["atol"]),
            )
            credit_recomputed = credit_recomputed and _credit_equal(recomputed, row)
        except (KeyError, TypeError, ValueError):
            evidence_valid = False
    nonzero = [
        row
        for row in rows
        if row.get("replay_valid") is True
        and row.get("decision") in {"rescue_preference", "reverse_preference"}
    ]
    decisions = Counter(str(row["decision"]) for row in nonzero)
    direction_tasks = {
        direction: {
            str(row["task_id_hash"])
            for row in nonzero
            if row["decision"] == direction
        }
        for direction in ("rescue_preference", "reverse_preference")
    }
    thresholds = protocol["thresholds"]
    bootstrap = _task_bootstrap(
        nonzero,
        int(thresholds["task_bootstrap_replicates"]),
        int(protocol["seed"]) + 9901,
    )
    total = len(nonzero)
    always_b = decisions["rescue_preference"] / max(total, 1)
    always_a = decisions["reverse_preference"] / max(total, 1)
    invalid_pair_rate = (len(rows) - int(audit.get("valid_pairs", 0))) / max(len(rows), 1)
    protocol_sha = file_sha256(args.protocol_lock)
    source_hashes = protocol.get("source_sha256", {})
    root = Path(__file__).resolve().parents[1]
    integrity = {
        "protocol_bound": audit.get("protocol_lock_sha256") == protocol_sha,
        "event_file_bound": audit.get("event_file_sha256") == file_sha256(args.raw_events),
        "protocol_validated": audit.get("status") == "completed"
        and audit.get("protocol_validated") is True,
        "collection_config_bound": audit.get("role") == "full"
        and audit.get("scenarios") == protocol.get("limit")
        and audit.get("candidate_count") == protocol.get("candidate_count")
        and audit.get("max_pairs_per_scenario") == protocol.get("max_pairs_per_scenario")
        and audit.get("horizon") == protocol.get("horizon")
        and audit.get("harness_interface") == protocol.get("harness_interface")
        and audit.get("credit_mode") == protocol.get("credit_mode"),
        "selected_scenarios_bound": audit.get("selected_scenario_hashes") == sealed_hashes
        and len(sealed_hashes) == protocol.get("limit"),
        "scenario_task_identity": identity_valid,
        "event_identity": len(event_ids) == len(set(event_ids)),
        "event_id_derivation": event_derivation_valid,
        "replay_validity_recomputed": replay_identity_valid,
        "scenario_pair_cap": all(
            count <= int(protocol["max_pairs_per_scenario"])
            for count in scenario_counts.values()
        ),
        "exact_snapshot": audit.get("snapshot_audit", {}).get("exact") is True,
        "runtime_bound": audit.get("toolsandbox_runtime") == protocol.get("toolsandbox_runtime"),
        "worker_bound": audit.get("worker_script_sha256")
        == source_hashes.get("scripts/toolsandbox_azure_worker.py"),
        "source_identity": bool(source_hashes)
        and all(
            (root / path).is_file() and file_sha256(root / path) == expected
            for path, expected in source_hashes.items()
        ),
        "untouched_hashes": not protocol.get("scenario_identity", {}).get(
            "historical_overlap"
        )
        and protocol.get("labels_inspected_before_freeze") is False
        and protocol.get("preexisting_offset205_artifacts") == [],
        "official_branch_evidence": evidence_valid,
        "credit_recomputed": credit_recomputed,
        "audit_counts_recomputed": audit.get("valid_pairs")
        == sum(row.get("replay_valid") is True for row in rows)
        and audit.get("nonzero_pairs") == total
        and audit.get("decisions")
        == dict(
            sorted(
                Counter(
                    str(row.get("decision"))
                    for row in rows
                    if row.get("replay_valid") is True
                ).items()
            )
        ),
    }
    outcomes = {
        "minimum_nonzero": total >= thresholds["min_valid_nonzero_events"],
        "minimum_rescue_events": decisions["rescue_preference"]
        >= thresholds["min_events_per_direction"],
        "minimum_reverse_events": decisions["reverse_preference"]
        >= thresholds["min_events_per_direction"],
        "minimum_rescue_tasks": len(direction_tasks["rescue_preference"])
        >= thresholds["min_tasks_per_direction"],
        "minimum_reverse_tasks": len(direction_tasks["reverse_preference"])
        >= thresholds["min_tasks_per_direction"],
        "task_bootstrap_both_directions": bootstrap[
            "probability_both_directions_observed"
        ]
        >= thresholds["min_task_bootstrap_direction_probability"],
        "invalid_pair_rate": invalid_pair_rate <= thresholds["max_invalid_pair_rate"],
        "worker_failure_rate": float(audit.get("worker_failure_rate", 1.0))
        <= thresholds["max_worker_failure_rate"],
    }
    passed = all(integrity.values()) and all(outcomes.values())
    result = {
        "passed": passed,
        "fresh_compensation_trap_supported": passed,
        "stage": "toolsandbox_compensation_trap_fresh_confirmation_gate",
        "integrity_checks": integrity,
        "outcome_checks": outcomes,
        "observed": {
            "scenarios": audit.get("scenarios"),
            "pairs": len(rows),
            "valid_pairs": audit.get("valid_pairs"),
            "invalid_pair_rate": invalid_pair_rate,
            "worker_failures": audit.get("worker_failures"),
            "worker_failure_rate": audit.get("worker_failure_rate"),
            "nonzero_events": total,
            "decision_counts": dict(sorted(decisions.items())),
            "direction_tasks": {
                key: len(value) for key, value in direction_tasks.items()
            },
            "always_a_accuracy": always_a,
            "always_b_accuracy": always_b,
            "best_constant_accuracy": max(always_a, always_b),
            "oracle_router_accuracy": 1.0 if total else None,
            "oracle_headroom_over_best_constant": 1.0 - max(always_a, always_b)
            if total
            else None,
            "task_bootstrap": bootstrap,
        },
        "thresholds": thresholds,
        "protocol_lock_sha256": protocol_sha,
        "audit_summary_sha256": file_sha256(args.audit_summary),
        "claim_boundary": protocol["claim_boundary"],
        "next_step": (
            "add the fresh row to the main Compensation Trap table"
            if passed
            else "report that the fixed untouched tail did not replicate both directions"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
