#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import digest, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_bounded import (
    CONFIRMATORY_CODE_PATHS,
    EXPECTED_EVENTS,
    EXPECTED_EVENT_FILE_SHA256,
    EXPECTED_EVENT_SET_HASH,
    EXPECTED_HORIZONS,
    summarize_horizon,
)
from rescuecredit.route_a_task_eval import event_set_hash


SEEDS = (43, 44, 45)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_717
TOLERANCE = 1e-12


def current_code_identity() -> dict[str, Any]:
    files = {path: file_sha256(Path(path)) for path in CONFIRMATORY_CODE_PATHS}
    return {"files": files, "fingerprint": digest(files)}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_score(row: dict[str, Any], method: str) -> float:
    payload = row["horizons"]["8"]
    selected = row[f"{method}_selected"]
    return float(payload["score_b"] if selected == "b" else payload["score_a"])


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def cluster_bootstrap(event_deltas: dict[str, list[float]]) -> dict[str, float]:
    clusters = sorted(event_deltas)
    cluster_means = {
        event_id: sum(event_deltas[event_id]) / len(event_deltas[event_id])
        for event_id in clusters
    }
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [cluster_means[rng.choice(clusters)] for _ in clusters]
        draws.append(sum(sample) / len(sample))
    point = sum(cluster_means.values()) / len(cluster_means)
    return {
        "mean": point,
        "ci95_lower": percentile(draws, 0.025),
        "ci95_upper": percentile(draws, 0.975),
        "clusters": len(clusters),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    analyzer_code_identity = current_code_identity()

    per_seed = {}
    event_deltas: dict[str, list[float]] = defaultdict(list)
    integrity_checks = {}
    valid_event_sets: dict[str, set[str]] = {}
    shared_identities: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        seed_root = args.root / f"seed{seed}"
        summary_path = seed_root / "bounded_summary.json"
        result_path = seed_root / "bounded_results.jsonl"
        lock_path = seed_root / "protocol_lock.json"
        summary = read_json(summary_path)
        rows = read_jsonl(result_path)
        lock = read_json(lock_path)
        reported_primary = summary["primary"]
        primary = summarize_horizon(rows, 8)
        row_ids = [str(row["event_id"]) for row in rows]
        valid_event_sets[str(seed)] = {
            str(row["event_id"])
            for row in rows
            if row.get("horizons", {}).get("8", {}).get("evaluation_valid")
        }
        shared_identities[str(seed)] = {
            "code_identity": summary.get("code_identity"),
            "policy_identity": summary.get("policy_identity"),
            "runtime_identity": summary.get("runtime_identity"),
        }
        checks = {
            "summary_seed": summary.get("seed") == seed,
            "confirmatory_flag": summary.get("confirmatory") is True,
            "full_event_count": len(rows) == EXPECTED_EVENTS,
            "unique_event_ids": len(set(row_ids)) == len(row_ids),
            "horizons": tuple(summary.get("requested_horizons", []))
            == EXPECTED_HORIZONS,
            "event_identity": summary.get("event_set_hash")
            == EXPECTED_EVENT_SET_HASH
            and summary.get("event_file_sha256") == EXPECTED_EVENT_FILE_SHA256,
            "protocol_validated": summary.get("protocol_lock_validated") is True,
            "lock_bound": summary.get("protocol_lock_sha256")
            == file_sha256(lock_path),
            "embedded_lock_matches": summary.get("protocol_lock") == lock,
            "results_bound": summary.get("bounded_results_sha256")
            == file_sha256(result_path)
            and int(summary.get("bounded_results_rows", -1)) == len(rows),
            "results_cover_frozen_events": summary.get(
                "bounded_results_event_set_hash"
            )
            == EXPECTED_EVENT_SET_HASH
            and event_set_hash(rows) == EXPECTED_EVENT_SET_HASH,
            "lock_status": lock.get("status")
            == "frozen_before_confirmatory_outcomes",
            "lock_seed": lock.get("seed") == seed,
            "lock_checks": all(lock.get("checks", {}).values()),
            "code_identity_bound": summary.get("code_identity")
            == lock.get("code_identity")
            == analyzer_code_identity,
            "policy_identity_bound": summary.get("policy_identity")
            == lock.get("policy_identity"),
            "runtime_identity_bound": summary.get("runtime_identity")
            == lock.get("runtime_identity"),
            "primary_recomputed_from_rows": reported_primary == primary,
            "no_cache_conflicts": int(summary.get("cache_conflicts", -1)) == 0,
            "reference_free": summary.get(
                "continuation_input_excludes_evaluator_and_reference"
            )
            is True,
            "no_test_or_reference_suffix": summary.get("test_split_access") is False
            and summary.get("reference_suffix_used") is False,
            "worker_isolation_enforced": summary.get("worker_cwd_isolated")
            is True
            and summary.get("worker_sandbox_location")
            == "system_tmp_outside_appworld_root"
            and summary.get("worker_benchmark_root_in_environment") is False
            and "APPWORLD_ROOT"
            not in summary.get("worker_environment_allowlist", []),
            "prefix_consistent": int(summary.get("horizon_prefix_mismatches", -1))
            == 0
            and int(summary.get("horizon_prefix_unverifiable", -1)) == 0,
            "enough_valid_events": int(primary["valid_paired_events"]) >= 40,
            "enough_nonzero_events": int(primary["nonzero_causal_events"]) >= 5,
        }
        integrity_checks[str(seed)] = checks
        if not all(checks.values()):
            raise ValueError(f"confirmatory integrity failure seed {seed}: {checks}")

        for row in rows:
            payload = row["horizons"]["8"]
            if not payload.get("evaluation_valid"):
                continue
            event_deltas[str(row["event_id"])].append(
                selected_score(row, "v2") - selected_score(row, "mask")
            )
        per_seed[str(seed)] = {
            "score_improvement": float(primary["score_improvement"]),
            "causal_accuracy_improvement": float(
                primary["causal_accuracy_improvement"]
            ),
            "nonzero_causal_events": int(primary["nonzero_causal_events"]),
            "v2_better_events": int(primary["v2_better_events"]),
            "v2_worse_events": int(primary["v2_worse_events"]),
            "ties": int(primary["ties"]),
            "wall_time_sec": float(summary["wall_time_sec"]),
        }

    identity_values = list(shared_identities.values())
    cross_seed_identity_match = all(
        identity == identity_values[0] for identity in identity_values[1:]
    )
    valid_sets = list(valid_event_sets.values())
    same_valid_event_subset = all(
        event_ids == valid_sets[0] for event_ids in valid_sets[1:]
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
    checks = {
        "all_integrity_checks_pass": all(
            all(seed_checks.values()) for seed_checks in integrity_checks.values()
        ),
        "cross_seed_code_policy_runtime_identity_match": cross_seed_identity_match,
        "same_valid_event_subset_across_seeds": same_valid_event_subset,
        "at_least_two_positive_seeds": positive_seeds >= 2,
        "enough_total_nonzero_events": total_nonzero >= 15,
        "positive_mean_score_improvement": sum(score_improvements) / len(SEEDS)
        > TOLERANCE,
        "positive_mean_causal_accuracy_improvement": sum(accuracy_improvements)
        / len(SEEDS)
        > TOLERANCE,
        "aggregate_wins_exceed_losses": total_wins > total_losses,
        "cluster_bootstrap_ci_lower_above_zero": bootstrap["ci95_lower"]
        > TOLERANCE,
    }
    result = {
        "status": "completed",
        "stage": "route_a_appworld_bounded_confirm_seeds_43_44_45",
        "passed": all(checks.values()),
        "statistically_supported": checks[
            "cluster_bootstrap_ci_lower_above_zero"
        ],
        "checks": checks,
        "per_seed": per_seed,
        "integrity_checks": integrity_checks,
        "valid_event_counts": {
            seed: len(event_ids) for seed, event_ids in valid_event_sets.items()
        },
        "mean_score_improvement": sum(score_improvements) / len(SEEDS),
        "mean_causal_accuracy_improvement": sum(accuracy_improvements)
        / len(SEEDS),
        "positive_seeds": positive_seeds,
        "total_nonzero_causal_events": total_nonzero,
        "aggregate_v2_wins": total_wins,
        "aggregate_v2_losses": total_losses,
        "cluster_bootstrap": bootstrap,
        "scope": (
            "confirmatory controlled-state bounded-horizon AppWorld causal diagnostic; "
            "not fully autonomous task success"
        ),
        "next_step": (
            "freeze the method and run the untouched final evaluation"
            if all(checks.values())
            else "do not access test; improve causal coverage or method on train only"
        ),
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
