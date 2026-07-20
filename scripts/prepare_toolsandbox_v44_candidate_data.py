#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from environments.toolsandbox import action_schema_complete, canonical_action
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash, public_preference_prompt
from scripts.freeze_toolsandbox_v44_candidate_protocol import (
    FULL_THRESHOLDS,
    PROTOCOL_STATUS,
)
from scripts.prepare_toolsandbox_v41_preference_data import (
    _branch_metrics,
    _relevant_schemas,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--data-role", choices=("train", "evaluation"), default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.audit_root / "audit_summary.json"
    event_path = args.audit_root / "candidate_events.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("role") != "full":
        raise RuntimeError("candidate data requires the frozen full V4.4 protocol")
    if protocol.get("thresholds") != FULL_THRESHOLDS:
        raise RuntimeError("V4.4 thresholds differ from the frozen protocol")
    if summary.get("status") != "completed" or summary.get("role") != "full":
        raise RuntimeError("full V4.4 candidate audit did not complete")
    if summary.get("protocol_validated") is not True:
        raise RuntimeError("V4.4 audit did not validate its protocol")
    if summary.get("protocol_lock_sha256") != file_sha256(args.protocol_lock):
        raise RuntimeError("V4.4 audit and protocol identity differ")
    if summary.get("event_file_sha256") != file_sha256(event_path):
        raise RuntimeError("V4.4 candidate event file identity differs")
    if summary.get("snapshot_audit", {}).get("exact") is not True:
        raise RuntimeError("V4.4 snapshot audit is not exact")

    horizon = int(summary["horizon"])
    source_rows = [
        row
        for row in read_jsonl(event_path)
        if row.get("mode") == "both_valid_candidate_pair"
        and row.get("replay_valid") is True
        and row.get("decision") in {"rescue_preference", "reverse_preference"}
    ]
    event_ids = [str(row["event_id"]) for row in source_rows]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("V4.4 event identifiers are not unique")

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for row in source_rows:
        history = row.get("treatment_visible_history")
        schemas = row.get("treatment_public_tool_schemas")
        if not isinstance(history, list) or not isinstance(schemas, list):
            raise ValueError("V4.4 event lacks visible treatment context")
        action_a = canonical_action(row["action_a"])
        action_b = canonical_action(row["action_b"])
        if action_a == action_b:
            raise ValueError("V4.4 candidates are identical")
        if not action_schema_complete(action_a, schemas) or not action_schema_complete(
            action_b, schemas
        ):
            raise ValueError("V4.4 candidate pair is not schema-complete")
        relevant = _relevant_schemas(schemas, action_a, action_b)
        public = {
            "event_id": str(row["event_id"]),
            "task_id_hash": str(row["task_id_hash"]),
            "mode": "both_valid_candidate_pair",
            "reference_free_prefix_steps": int(row["reference_free_prefix_steps"]),
            "candidate_rank": int(row["candidate_rank"]),
            "prompt": public_preference_prompt(
                visible_history=history,
                public_tool_schemas=relevant,
                action_a=action_a,
                action_b=action_b,
            ),
            "action_a": action_a,
            "action_b": action_b,
        }
        private = {
            "event_id": str(row["event_id"]),
            "replay_valid": True,
            "decision": str(row["decision"]),
            "decision_basis": str(row["decision_basis"]),
            "decision_value": float(row["decision_value"]),
            "causal_weight": float(row["causal_weight"]),
            "branch_a": _branch_metrics(row["branch_a"], horizon),
            "branch_b": _branch_metrics(row["branch_b"], horizon),
        }
        public_rows.append(public)
        private_rows.append(private)
        train_rows.append(
            {
                **public,
                "replay_valid": True,
                "decision": private["decision"],
                "decision_basis": private["decision_basis"],
                "causal_weight": private["causal_weight"],
            }
        )

    public_rows.sort(key=lambda row: row["event_id"])
    private_rows.sort(key=lambda row: row["event_id"])
    train_rows.sort(key=lambda row: row["event_id"])
    decisions = Counter(str(row["decision"]) for row in private_rows)
    task_counts = Counter(str(row["task_id_hash"]) for row in public_rows)
    public_by_id = {str(row["event_id"]): row for row in public_rows}
    direction_tasks = {
        direction: {
            str(public_by_id[str(private["event_id"])]["task_id_hash"])
            for private in private_rows
            if private["decision"] == direction
        }
        for direction in ("rescue_preference", "reverse_preference")
    }
    max_task_pairs = max(task_counts.values(), default=0)
    max_task_share = max_task_pairs / max(1, len(public_rows))
    checks = {
        "enough_scenarios": int(summary["scenarios"])
        >= FULL_THRESHOLDS["min_scenarios"],
        "enough_valid_pairs": int(summary["valid_pairs"])
        >= FULL_THRESHOLDS["min_valid_pairs"],
        "enough_nonzero_pairs": len(public_rows)
        >= FULL_THRESHOLDS["min_nonzero_pairs"],
        "enough_rescue_pairs": decisions.get("rescue_preference", 0)
        >= FULL_THRESHOLDS["min_rescue_pairs"],
        "enough_reverse_pairs": decisions.get("reverse_preference", 0)
        >= FULL_THRESHOLDS["min_reverse_pairs"],
        "enough_rescue_tasks": len(direction_tasks["rescue_preference"])
        >= FULL_THRESHOLDS["min_rescue_tasks"],
        "enough_reverse_tasks": len(direction_tasks["reverse_preference"])
        >= FULL_THRESHOLDS["min_reverse_tasks"],
        "task_pair_cap": max_task_pairs
        <= FULL_THRESHOLDS["max_pairs_per_task"],
        "not_task_dominated": max_task_share
        <= FULL_THRESHOLDS["max_task_pair_share"],
        "worker_failure_rate": float(summary["worker_failure_rate"])
        <= FULL_THRESHOLDS["max_worker_failure_rate"],
        "snapshot_restore_exact": summary.get("snapshot_audit", {}).get("exact")
        is True,
        "both_candidates_public_schema_valid": all(
            row["mode"] == "both_valid_candidate_pair" for row in public_rows
        ),
        "private_outcomes_absent_from_training": all(
            "branch_a" not in row and "branch_b" not in row for row in train_rows
        ),
    }
    passed = all(checks.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_path = args.output_dir / "events.public.jsonl"
    private_path = args.output_dir / "outcomes.private.jsonl"
    train_path = args.output_dir / "train.jsonl"
    write_jsonl(public_path, public_rows)
    write_jsonl(private_path, private_rows)
    write_jsonl(train_path, train_rows)
    manifest = {
        "status": "frozen" if passed else "rejected",
        "stage": "toolsandbox_v44_candidate_diversity_data",
        "role": args.data_role,
        "passed": passed,
        "events": len(public_rows),
        "tasks": len(task_counts),
        "decisions": dict(sorted(decisions.items())),
        "direction_tasks": {
            key: len(value) for key, value in direction_tasks.items()
        },
        "max_task_pairs": max_task_pairs,
        "max_task_pair_share": max_task_share,
        "event_set_hash": event_set_hash(public_rows),
        "source_event_sha256": file_sha256(event_path),
        "source_summary_sha256": file_sha256(summary_path),
        "source_protocol_sha256": file_sha256(args.protocol_lock),
        "public_sha256": file_sha256(public_path),
        "private_sha256": file_sha256(private_path),
        "train_sha256": file_sha256(train_path),
        "checks": checks,
        "thresholds": FULL_THRESHOLDS,
        "candidate_generation": "reference_free_unranked_schema_valid_alternatives",
        "official_branch_metrics_in_training_file": False,
        "protected_outcomes_in_prompt": False,
        "branch_receipts_exported": False,
        "reference_actions_read_or_exported": False,
        "next_step": (
            "freeze the matched anchored learner protocol"
            if passed
            else "stop before training; candidate diversity remains insufficient"
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(
        args.output_dir / "data_gate.json",
        {
            "passed": passed,
            "stage": "toolsandbox_v44_candidate_diversity_data_gate",
            "checks": checks,
            "thresholds": FULL_THRESHOLDS,
            "events": len(public_rows),
            "decisions": dict(sorted(decisions.items())),
            "direction_tasks": {
                key: len(value) for key, value in direction_tasks.items()
            },
            "next_step": manifest["next_step"],
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
