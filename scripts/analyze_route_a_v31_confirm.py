#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_bounded import summarize_horizon
from rescuecredit.route_a_task_eval import event_set_hash
from scripts.freeze_route_a_v31_confirm_protocol import (
    AGGREGATE_THRESHOLDS,
    CONFIRMATORY_SEEDS,
)


TOLERANCE = 1e-12


def equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= TOLERANCE
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            equivalent(a, b) for a, b in zip(left, right)
        )
    return left == right


def selected_score(row: dict[str, Any], method: str) -> float:
    payload = row["horizons"]["8"]
    selected = row[f"{method}_selected"]
    return float(payload["score_b"] if selected == "b" else payload["score_a"])


def cluster_bootstrap(event_deltas: dict[str, list[float]]) -> dict[str, float | int]:
    clusters = sorted(event_deltas)
    means = {key: sum(event_deltas[key]) / len(event_deltas[key]) for key in clusters}
    rng = random.Random(20260718)
    draws: list[float] = []
    for _ in range(10_000):
        sample = [means[rng.choice(clusters)] for _ in clusters]
        draws.append(sum(sample) / len(sample))
    ordered = sorted(draws)
    return {
        "mean": sum(means.values()) / len(means),
        "ci95_lower": ordered[round(0.025 * (len(ordered) - 1))],
        "ci95_upper": ordered[round(0.975 * (len(ordered) - 1))],
        "clusters": len(clusters),
        "samples": len(draws),
        "seed": 20260718,
        "gating_role": "reported_not_gated",
    }


def analyze(root: Path) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    all_checks: dict[str, dict[str, bool]] = {}
    task_deltas: dict[str, list[float]] = defaultdict(list)
    event_hashes: set[str] = set()
    valid_event_sets: dict[str, set[str]] = {}
    for seed in CONFIRMATORY_SEEDS:
        seed_root = root / f"seed{seed}"
        load = lambda name: json.loads((seed_root / name).read_text(encoding="utf-8"))
        training_lock = load("training_protocol_lock.json")
        preference_gate = load("preference_gate.json")
        bounded_lock = load("bounded_protocol_lock.json")
        summary = load("bounded_summary.json")
        result_path = seed_root / "bounded_results.jsonl"
        rows = read_jsonl(result_path)
        primary = summarize_horizon(rows, 8)
        row_hash = event_set_hash(rows)
        event_hashes.add(row_hash)
        valid_event_sets[str(seed)] = {
            str(row["event_id"])
            for row in rows
            if row["horizons"]["8"].get("evaluation_valid")
        }
        training_source_ok = all(
            Path(path).is_file() and file_sha256(Path(path)) == expected
            for path, expected in training_lock.get("source_sha256", {}).items()
        )
        source_ok = all(
            Path(path).is_file() and file_sha256(Path(path)) == expected
            for path, expected in bounded_lock.get("source_sha256", {}).items()
        )
        checks = {
            "training_lock_seed": training_lock.get("seed") == seed,
            "training_lock_preoutcome": training_lock.get("status")
            == "frozen_before_v31_outcomes",
            "training_source_identity": bool(training_lock.get("source_sha256"))
            and training_source_ok,
            "aggregate_thresholds_frozen": training_lock.get("aggregate_thresholds")
            == bounded_lock.get("aggregate_thresholds") == AGGREGATE_THRESHOLDS,
            "preference_integrity_passed": preference_gate.get("passed") is True,
            "bounded_lock_preoutcome": bounded_lock.get("status")
            == "frozen_before_both_valid_confirmatory_outcomes",
            "bounded_lock_seed": bounded_lock.get("seed") == seed,
            "evaluator_validated_lock": summary.get("protocol_lock_validated") is True
            and summary.get("development_confirmatory") is True
            and summary.get("protocol_lock_sha256")
            == file_sha256(seed_root / "bounded_protocol_lock.json"),
            "raw_summary_recomputed": equivalent(summary.get("primary"), primary),
            "result_identity": summary.get("bounded_results_sha256")
            == file_sha256(result_path)
            and summary.get("bounded_results_event_set_hash") == row_hash
            == bounded_lock.get("event_set_hash"),
            "full_run": summary.get("sanity_limit") is None
            and len(rows) == int(bounded_lock.get("events", -1)),
            "reference_boundary": summary.get(
                "continuation_input_excludes_evaluator_and_reference"
            )
            is True
            and summary.get("reference_suffix_used") is False
            and summary.get("test_split_access") is False,
            "cache_and_prefix_integrity": int(summary.get("cache_conflicts", -1)) == 0
            and int(summary.get("horizon_prefix_mismatches", -1)) == 0
            and int(summary.get("horizon_prefix_unverifiable", -1)) == 0,
            "source_identity": bool(bounded_lock.get("source_sha256")) and source_ok,
            "enough_valid_events": int(primary["valid_paired_events"]) >= 30,
        }
        all_checks[str(seed)] = checks
        for row in rows:
            if not row["horizons"]["8"].get("evaluation_valid"):
                continue
            task_deltas[str(row["task_id_hash"])].append(
                selected_score(row, "v2") - selected_score(row, "mask")
            )
        per_seed[str(seed)] = {
            "preference_gate_passed": preference_gate.get("passed") is True,
            "preference_both_valid_accuracy_improvement": preference_gate[
                "both_valid_accuracy_improvement"
            ],
            "valid_paired_events": primary["valid_paired_events"],
            "nonzero_causal_events": primary["nonzero_causal_events"],
            "score_improvement": primary["score_improvement"],
            "causal_accuracy_improvement": primary["causal_accuracy_improvement"],
            "v31_better_events": primary["v2_better_events"],
            "v31_worse_events": primary["v2_worse_events"],
            "ties": primary["ties"],
        }

    scores = [float(row["score_improvement"]) for row in per_seed.values()]
    causal = [float(row["causal_accuracy_improvement"]) for row in per_seed.values()]
    positive_seeds = sum(value > TOLERANCE for value in scores)
    total_nonzero = sum(int(row["nonzero_causal_events"]) for row in per_seed.values())
    wins = sum(int(row["v31_better_events"]) for row in per_seed.values())
    losses = sum(int(row["v31_worse_events"]) for row in per_seed.values())
    checks = {
        "all_seed_integrity_checks_pass": all(
            all(seed_checks.values()) for seed_checks in all_checks.values()
        ),
        "same_frozen_event_set": len(event_hashes) == 1,
        "same_valid_event_subset": len(
            {frozenset(event_ids) for event_ids in valid_event_sets.values()}
        )
        == 1,
        "minimum_positive_score_seeds": positive_seeds
        >= AGGREGATE_THRESHOLDS["minimum_positive_score_seeds"],
        "minimum_total_nonzero_events": total_nonzero
        >= AGGREGATE_THRESHOLDS["minimum_total_nonzero_events"],
        "positive_mean_score_improvement": sum(scores) / len(scores) > TOLERANCE,
        "positive_mean_causal_accuracy_improvement": sum(causal) / len(causal)
        > TOLERANCE,
        "aggregate_wins_over_losses": wins > losses,
    }
    return {
        "status": "completed",
        "stage": "route_a_v31_confirm_seeds_43_44_45",
        "passed": all(checks.values()),
        "checks": checks,
        "per_seed": per_seed,
        "seed_integrity_checks": all_checks,
        "positive_score_seeds": positive_seeds,
        "total_nonzero_causal_events": total_nonzero,
        "mean_score_improvement": sum(scores) / len(scores),
        "mean_causal_accuracy_improvement": sum(causal) / len(causal),
        "aggregate_v31_wins": wins,
        "aggregate_v31_losses": losses,
        "cluster_bootstrap": {
            **cluster_bootstrap(task_deltas),
            "cluster_unit": "task_id_hash",
        },
        "scope": "training-seed and continuation-seed confirmation on the same frozen both-valid AppWorld dev fixture; not fresh-task or autonomous success evidence",
        "next_step": "retain AppWorld as replicated secondary-environment evidence and begin ToolSandbox signal audit",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
