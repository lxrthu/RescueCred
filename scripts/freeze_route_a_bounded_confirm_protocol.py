#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_bounded import (
    EXPECTED_EVENTS,
    EXPECTED_EVENT_FILE_SHA256,
    EXPECTED_EVENT_SET_HASH,
    EXPECTED_HORIZONS,
)
from rescuecredit.route_a_task_eval import event_set_hash
from evaluate_route_a_bounded import (
    _confirmatory_code_identity,
    _policy_identity,
    _runtime_identity,
)


CONFIRMATORY_SEEDS = (43, 44, 45)


def _ids(path: Path) -> list[str]:
    return [str(row["event_id"]) for row in read_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("--mask-results", type=Path, required=True)
    parser.add_argument("--v2-results", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=CONFIRMATORY_SEEDS, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.appworld_root.resolve()
    from appworld import AppWorld, update_root

    update_root(str(root))

    events = read_jsonl(args.event_file)
    event_ids = [str(row["event_id"]) for row in events]
    checks = {
        "exact_event_count": len(event_ids) == EXPECTED_EVENTS,
        "unique_event_ids": len(set(event_ids)) == len(event_ids),
        "exact_event_set_hash": event_set_hash(events) == EXPECTED_EVENT_SET_HASH,
        "exact_event_file_sha256": file_sha256(args.event_file)
        == EXPECTED_EVENT_FILE_SHA256,
        "mask_covers_exact_event_set": set(_ids(args.mask_results))
        == set(event_ids)
        and len(_ids(args.mask_results)) == len(event_ids),
        "v2_covers_exact_event_set": set(_ids(args.v2_results)) == set(event_ids)
        and len(_ids(args.v2_results)) == len(event_ids),
        "dev_fixture_only": all(
            row.get("split") == "dev" and isinstance(row.get("task_index"), int)
            for row in events
        ),
    }
    if not all(checks.values()):
        raise SystemExit(json.dumps({"passed": False, "checks": checks}, indent=2))

    code_identity = _confirmatory_code_identity()
    policy_identity = _policy_identity(args.worker_script)
    runtime_identity = _runtime_identity(
        root=root, AppWorld=AppWorld, events=events, seed=args.seed
    )

    lock = {
        "status": "frozen_before_confirmatory_outcomes",
        "stage": f"route_a_appworld_bounded_confirm_seed{args.seed}",
        "seed": args.seed,
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "horizons": list(EXPECTED_HORIZONS),
        "events": EXPECTED_EVENTS,
        "event_set_hash": EXPECTED_EVENT_SET_HASH,
        "event_file_sha256": file_sha256(args.event_file),
        "mask_results_sha256": file_sha256(args.mask_results),
        "v2_results_sha256": file_sha256(args.v2_results),
        "checks": checks,
        "code_identity": code_identity,
        "policy_identity": policy_identity,
        "runtime_identity": runtime_identity,
        "test_split_access": False,
        "gate_frozen_before_outcomes": {
            "minimum_positive_seeds": 2,
            "minimum_total_nonzero_events": 15,
            "require_positive_mean_score_improvement": True,
            "require_positive_mean_causal_accuracy_improvement": True,
            "require_aggregate_wins_over_losses": True,
            "require_cluster_bootstrap_ci_lower_above_zero": True,
            "bootstrap_clusters": "event_id",
            "bootstrap_samples": 10000,
            "bootstrap_seed": 20260717,
        },
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != lock:
            raise SystemExit("existing confirmatory protocol lock differs")
    else:
        write_json(args.output, lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
