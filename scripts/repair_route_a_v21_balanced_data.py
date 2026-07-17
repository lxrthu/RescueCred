#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json


def _counts(rows: list[dict]) -> dict:
    decisions = Counter(str(row["decision"]) for row in rows)
    causal = [row for row in rows if row["decision"] != "zero_delta"]
    by_task = Counter(str(row["task_id"]) for row in causal)
    return {
        "events": len(rows),
        "tasks": len({str(row["task_id"]) for row in rows}),
        "nonzero_events": len(causal),
        "nonzero_tasks": len(by_task),
        "max_nonzero_events_per_task": max(by_task.values(), default=0),
        "max_task_nonzero_share": max(by_task.values(), default=0)
        / max(1, len(causal)),
        "decisions": dict(decisions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair V2.1 data using outcome-independent replay and task caps"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-events-per-task", type=int, default=10)
    args = parser.parse_args()
    if args.max_events_per_task <= 0:
        raise ValueError("--max-events-per-task must be positive")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")

    started = time.time()
    source_bank = args.source_root / "bank"
    source_shadow = args.source_root / "shadow"
    source_dense = args.source_root / "dense"
    public_path = source_bank / "correction_bank.public.jsonl"
    private_path = source_bank / "offline_audit.private.jsonl"
    binary_path = source_shadow / "shadow_credit.train.jsonl"
    dense_path = source_dense / "dense_shadow_credit.train.jsonl"
    source_shadow_summary = json.loads(
        (source_shadow / "shadow_summary.json").read_text(encoding="utf-8")
    )

    public = read_jsonl(public_path)
    private_by_id = {str(row["event_id"]): row for row in read_jsonl(private_path)}
    binary_by_id = {str(row["event_id"]): row for row in read_jsonl(binary_path)}
    dense_by_id = {str(row["event_id"]): row for row in read_jsonl(dense_path)}
    checkpoints = source_shadow / "event_checkpoints"

    replay_valid_ids: set[str] = set()
    invalid_reasons: Counter[str] = Counter()
    for event in public:
        event_id = str(event["event_id"])
        checkpoint_path = checkpoints / f"{event_id}.json"
        if not checkpoint_path.is_file():
            invalid_reasons["checkpoint_missing"] += 1
            continue
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("status") != "valid":
            invalid_reasons[str(checkpoint.get("reason") or "unknown")] += 1
            continue
        if event_id not in binary_by_id or event_id not in dense_by_id:
            invalid_reasons["credit_missing"] += 1
            continue
        replay_valid_ids.add(event_id)

    # The cap is deliberately applied to every replay-valid event before any
    # delta or preference label is read. This prevents outcome-based cherry-picking.
    by_task: dict[str, list[dict]] = defaultdict(list)
    for event in public:
        if str(event["event_id"]) in replay_valid_ids:
            by_task[str(event["task_id"])].append(event)
    selected_public: list[dict] = []
    for task_id in sorted(by_task):
        selected_public.extend(
            sorted(by_task[task_id], key=lambda row: str(row["event_id"]))[
                : args.max_events_per_task
            ]
        )
    selected_public.sort(key=lambda row: str(row["event_id"]))
    selected_ids = {str(row["event_id"]) for row in selected_public}
    selected_private = [private_by_id[event_id] for event_id in sorted(selected_ids)]
    selected_binary = [binary_by_id[event_id] for event_id in sorted(selected_ids)]
    selected_dense = [dense_by_id[event_id] for event_id in sorted(selected_ids)]

    if set(private_by_id) & selected_ids != selected_ids:
        raise AssertionError("private audit join is incomplete")
    if len(selected_ids) != len(selected_public):
        raise AssertionError("selected event ids are not unique")
    if any(len(rows) > args.max_events_per_task for rows in (
        [row for row in selected_public if str(row["task_id"]) == task]
        for task in by_task
    )):
        raise AssertionError("task cap was not enforced")

    bank_dir = args.output_root / "bank"
    shadow_dir = args.output_root / "shadow"
    dense_dir = args.output_root / "dense"
    bank_dir.mkdir(parents=True)
    shadow_dir.mkdir()
    dense_dir.mkdir()
    output_public = bank_dir / "correction_bank.public.jsonl"
    output_private = bank_dir / "offline_audit.private.jsonl"
    output_binary = shadow_dir / "shadow_credit.train.jsonl"
    output_dense = dense_dir / "dense_shadow_credit.train.jsonl"
    write_jsonl(output_public, selected_public)
    write_jsonl(output_private, selected_private)
    write_jsonl(output_binary, selected_binary)
    write_jsonl(output_dense, selected_dense)

    task_counts = Counter(str(row["task_id"]) for row in selected_public)
    bank_manifest = {
        "schema_version": "rescuecredit.route_a_bank.v2.1c",
        "status": "frozen",
        "split": "train",
        "tasks": 90,
        "tasks_with_events": len(task_counts),
        "events": len(selected_public),
        "max_events_per_task": max(task_counts.values(), default=0),
        "public_bank_sha256": file_sha256(output_public),
        "private_audit_sha256": file_sha256(output_private),
        "source_public_bank_sha256": file_sha256(public_path),
        "repair": {
            "rule": "replay-valid prefix then event-id cap per task",
            "max_events_per_task": args.max_events_per_task,
            "selection_reads_delta_or_preference": False,
            "source_events": len(public),
            "technical_invalid_events_excluded": len(public) - len(replay_valid_ids),
            "technical_invalid_reasons": dict(invalid_reasons),
            "balance_exclusions": len(replay_valid_ids) - len(selected_public),
        },
        "hard_boundary": {
            "training_reads": "correction_bank.public.jsonl only",
            "training_forbidden": "offline_audit.private.jsonl",
            "dev_or_test_events": 0,
        },
    }
    write_json(bank_dir / "manifest.json", bank_manifest)

    binary_decisions = Counter(str(row["decision"]) for row in selected_binary)
    shadow_summary = {
        "status": "completed",
        "requested_events": len(selected_public),
        "valid_events": len(selected_binary),
        "replay_valid_rate": len(selected_binary) / max(1, len(selected_public)),
        "nonzero_events": sum(abs(float(row["delta"])) > 1e-12 for row in selected_binary),
        "decisions": dict(binary_decisions),
        "failure_reasons": {},
        "source_replay_valid_rate": source_shadow_summary["replay_valid_rate"],
        "technical_invalid_events_excluded_before_freeze": len(public)
        - len(replay_valid_ids),
        "cache_conflicts": source_shadow_summary.get("cache_conflicts", 0),
        "worker_cwd_isolated": source_shadow_summary.get("worker_cwd_isolated"),
        "worker_benchmark_root_in_environment": source_shadow_summary.get(
            "worker_benchmark_root_in_environment"
        ),
        "offline_audit_private_read": False,
        "selection_reads_delta_or_preference": False,
        "source_shadow_summary_sha256": file_sha256(
            source_shadow / "shadow_summary.json"
        ),
    }
    write_json(shadow_dir / "shadow_summary.json", shadow_summary)

    dense_summary = {
        "status": "completed",
        **_counts(
            [
                {
                    **row,
                    "task_id": next(
                        event["task_id"]
                        for event in selected_public
                        if event["event_id"] == row["event_id"]
                    ),
                }
                for row in selected_dense
            ]
        ),
        "reward_source": "AppWorld official evaluator pass/(pass+fail)",
        "requirement_text_read_by_training": False,
        "offline_audit_private_read": False,
    }
    write_json(dense_dir / "dense_shadow_summary.json", dense_summary)

    repair_manifest = {
        "status": "frozen",
        "stage": "route_a_v21c_outcome_independent_repair",
        "selection_rule_frozen": {
            "technical_filter": "checkpoint status must be replay-valid",
            "technical_filter_uses_branch_return": False,
            "balance_order": "ascending public event_id",
            "max_events_per_task": args.max_events_per_task,
            "balance_rule_uses_delta_or_preference": False,
        },
        "source_events": len(public),
        "replay_valid_source_events": len(replay_valid_ids),
        "selected_events": len(selected_public),
        "selected_tasks": len(task_counts),
        "selected_dense": _counts(
            [
                {
                    **row,
                    "task_id": next(
                        event["task_id"]
                        for event in selected_public
                        if event["event_id"] == row["event_id"]
                    ),
                }
                for row in selected_dense
            ]
        ),
        "private_join_exact": {row["event_id"] for row in selected_public}
        == {row["event_id"] for row in selected_private},
        "public_bank_sha256": file_sha256(output_public),
        "private_audit_sha256": file_sha256(output_private),
        "binary_credit_sha256": file_sha256(output_binary),
        "dense_credit_sha256": file_sha256(output_dense),
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_root / "repair_manifest.json", repair_manifest)
    print(json.dumps(repair_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
