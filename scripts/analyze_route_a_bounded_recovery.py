#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from analyze_route_a_bounded_confirm import cluster_bootstrap, current_code_identity
except ModuleNotFoundError:
    from scripts.analyze_route_a_bounded_confirm import (
        cluster_bootstrap,
        current_code_identity,
    )
from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_bounded import (
    EXPECTED_EVENTS,
    EXPECTED_EVENT_SET_HASH,
    EXPECTED_HORIZONS,
    summarize_horizon,
)
from rescuecredit.route_a_task_eval import event_set_hash


SEEDS = (43, 44, 45)
TOLERANCE = 1e-12


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def numerically_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            numerically_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            numerically_equal(a, b) for a, b in zip(left, right)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def selected_score(row: dict[str, Any], method: str) -> float:
    payload = row["horizons"]["8"]
    return float(payload["score_b"] if row[f"{method}_selected"] == "b" else payload["score_a"])


def verifiable(row: dict[str, Any]) -> bool:
    return bool(
        row["horizons"]["4"]["evaluation_valid"]
        and row["horizons"]["8"]["evaluation_valid"]
        and row["horizon_prefix_match_a"]
        and row["horizon_prefix_match_b"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    code_identity = current_code_identity()
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    verifiable_ids: dict[int, set[str]] = {}
    integrity: dict[str, dict[str, bool]] = {}
    unverifiable_counts: dict[str, int] = {}

    for seed in SEEDS:
        seed_root = args.root / f"seed{seed}"
        summary_path = seed_root / "bounded_summary.json"
        results_path = seed_root / "bounded_results.jsonl"
        lock_path = seed_root / "protocol_lock.json"
        summary = read_json(summary_path)
        rows = read_jsonl(results_path)
        lock = read_json(lock_path)
        recomputed = summarize_horizon(rows, 8)
        ids = {str(row["event_id"]) for row in rows}
        eligible = {str(row["event_id"]) for row in rows if verifiable(row)}
        rows_by_seed[seed] = rows
        verifiable_ids[seed] = eligible
        unverifiable_counts[str(seed)] = sum(
            row["horizons"]["8"]["evaluation_valid"] and not verifiable(row)
            for row in rows
        )
        checks = {
            "seed_and_confirmatory": summary.get("seed") == seed
            and summary.get("confirmatory") is True,
            "full_unique_frozen_event_set": len(rows) == EXPECTED_EVENTS
            and len(ids) == len(rows)
            and event_set_hash(rows) == EXPECTED_EVENT_SET_HASH,
            "horizons": tuple(summary.get("requested_horizons", []))
            == EXPECTED_HORIZONS,
            "results_hash_bound": summary.get("bounded_results_sha256")
            == file_sha256(results_path),
            "lock_hash_bound": summary.get("protocol_lock_sha256")
            == file_sha256(lock_path),
            "embedded_lock_matches": summary.get("protocol_lock") == lock,
            "code_identity_still_frozen": summary.get("code_identity")
            == lock.get("code_identity")
            == code_identity,
            "policy_and_runtime_bound": summary.get("policy_identity")
            == lock.get("policy_identity")
            and summary.get("runtime_identity") == lock.get("runtime_identity"),
            "reported_primary_matches_rows_with_numeric_tolerance": numerically_equal(
                summary.get("primary"), recomputed
            ),
            "no_observed_prefix_mismatch": int(
                summary.get("horizon_prefix_mismatches", -1)
            )
            == 0,
            "worker_isolated": summary.get("worker_cwd_isolated") is True
            and summary.get("worker_benchmark_root_in_environment") is False,
            "no_cache_conflicts": int(summary.get("cache_conflicts", -1)) == 0,
            "reference_free_no_test": summary.get(
                "continuation_input_excludes_evaluator_and_reference"
            )
            is True
            and summary.get("reference_suffix_used") is False
            and summary.get("test_split_access") is False,
        }
        integrity[str(seed)] = checks
        if not all(checks.values()):
            raise ValueError(f"hard integrity failure seed {seed}: {checks}")

    common_ids = set.intersection(*(verifiable_ids[seed] for seed in SEEDS))
    per_seed: dict[str, dict[str, Any]] = {}
    event_deltas: dict[str, list[float]] = defaultdict(list)
    for seed in SEEDS:
        common_rows = [
            row for row in rows_by_seed[seed] if str(row["event_id"]) in common_ids
        ]
        primary = summarize_horizon(common_rows, 8)
        per_seed[str(seed)] = primary
        for row in common_rows:
            event_deltas[str(row["event_id"])].append(
                selected_score(row, "v2") - selected_score(row, "mask")
            )

    score_improvements = [row["score_improvement"] for row in per_seed.values()]
    accuracy_improvements = [
        row["causal_accuracy_improvement"] for row in per_seed.values()
    ]
    positive_seeds = sum(value > TOLERANCE for value in score_improvements)
    total_nonzero = sum(row["nonzero_causal_events"] for row in per_seed.values())
    total_wins = sum(row["v2_better_events"] for row in per_seed.values())
    total_losses = sum(row["v2_worse_events"] for row in per_seed.values())
    bootstrap = cluster_bootstrap(event_deltas)
    sensitivity_checks = {
        "at_least_40_common_verifiable_events": len(common_ids) >= 40,
        "at_least_two_positive_seeds": positive_seeds >= 2,
        "at_least_15_total_nonzero_events": total_nonzero >= 15,
        "positive_mean_score_improvement": sum(score_improvements) / len(SEEDS)
        > TOLERANCE,
        "positive_mean_causal_accuracy_improvement": sum(accuracy_improvements)
        / len(SEEDS)
        > TOLERANCE,
        "aggregate_wins_exceed_losses": total_wins > total_losses,
        "cluster_bootstrap_ci_lower_above_zero": bootstrap["ci95_lower"]
        > TOLERANCE,
    }
    original_prefix_verifiability_passed = all(
        count == 0 for count in unverifiable_counts.values()
    )
    result = {
        "status": "completed",
        "stage": "route_a_appworld_bounded_posthoc_common_verifiable_sensitivity",
        "original_preregistered_confirmatory_gate_passed": False,
        "original_prefix_verifiability_passed": original_prefix_verifiability_passed,
        "original_failure_reason": (
            None
            if original_prefix_verifiability_passed
            else "some H8-valid events lacked a verifiable H4 prefix"
        ),
        "posthoc_sensitivity_passed": all(sensitivity_checks.values()),
        "sensitivity_checks": sensitivity_checks,
        "hard_integrity_checks": integrity,
        "unverifiable_primary_events_per_seed": unverifiable_counts,
        "common_verifiable_events": len(common_ids),
        "per_seed_common_subset": per_seed,
        "positive_seeds": positive_seeds,
        "mean_score_improvement": sum(score_improvements) / len(SEEDS),
        "mean_causal_accuracy_improvement": sum(accuracy_improvements) / len(SEEDS),
        "total_nonzero_causal_events": total_nonzero,
        "aggregate_v2_wins": total_wins,
        "aggregate_v2_losses": total_losses,
        "cluster_bootstrap": bootstrap,
        "scope": (
            "post-hoc common-verifiable-event sensitivity analysis; it does not "
            "replace the failed preregistered confirmatory gate"
        ),
        "next_step": (
            "freeze a retry/missingness rule and run a new confirmation"
            if all(sensitivity_checks.values())
            else "do not access test; the causal advantage is not robust"
        ),
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
