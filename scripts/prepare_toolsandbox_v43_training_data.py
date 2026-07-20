#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from environments.toolsandbox import canonical_action
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import (
    NONZERO_DECISIONS,
    event_set_hash,
    public_preference_prompt,
)
from scripts.prepare_toolsandbox_v41_preference_data import (
    _branch_metrics,
    _relevant_schemas,
)


THRESHOLDS = {
    "min_events": 60,
    "min_reverse_events": 8,
    "min_reverse_tasks": 5,
    "max_task_event_share": 0.10,
    "max_events_per_task": 4,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.audit_root / "audit_summary.json"
    gate_path = args.audit_root / "quality_gate.json"
    signal_path = args.audit_root / "signal_events.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise RuntimeError("multi-prefix audit did not complete")
    if summary.get("credit_mode") != "lexicographic_v4":
        raise RuntimeError("training audit does not use V4 credit")
    if summary.get("harness_interface") != "tool_id_v2":
        raise RuntimeError("training audit does not use Tool-ID")
    if summary.get("protocol_validated") is not True:
        raise RuntimeError("training audit protocol was not validated")
    if summary.get("max_events_per_scenario") != 4:
        raise RuntimeError("training audit is not the frozen four-prefix run")
    if summary.get("event_file_sha256") != file_sha256(signal_path):
        raise RuntimeError("training signal file does not match its summary")
    if audit_gate.get("mechanism_passed") is not True:
        raise RuntimeError("multi-prefix mechanism gate did not pass")
    snapshot = summary.get("snapshot_audit", {})
    if snapshot.get("exact") is not True:
        raise RuntimeError("multi-prefix snapshot audit is not exact")

    horizon = int(summary["horizon"])
    source_rows = [
        row
        for row in read_jsonl(signal_path)
        if row.get("replay_valid") is True
        and row.get("decision") in NONZERO_DECISIONS
        and row.get("mode")
        in {"controlled_missing_argument", "natural_visible_error_repair"}
    ]
    ids = [str(row["event_id"]) for row in source_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("multi-prefix event ids are not unique")

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for row in source_rows:
        visible_history = row.get("treatment_visible_history")
        schemas = row.get("treatment_public_tool_schemas")
        if not isinstance(visible_history, list) or not isinstance(schemas, list):
            raise ValueError("multi-prefix row lacks its deployment-visible prefix")
        action_a = canonical_action(row["action_a"])
        action_b = canonical_action(row["action_b"])
        relevant = _relevant_schemas(schemas, action_a, action_b)
        public = {
            "event_id": str(row["event_id"]),
            "task_id_hash": str(row["task_id_hash"]),
            "mode": str(row["mode"]),
            "reference_free_prefix_steps": int(row["reference_free_prefix_steps"]),
            "prompt": public_preference_prompt(
                visible_history=visible_history,
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
    decisions = Counter(row["decision"] for row in private_rows)
    tasks = Counter(row["task_id_hash"] for row in public_rows)
    public_by_id = {str(row["event_id"]): row for row in public_rows}
    reverse_tasks = {
        public_by_id[str(private["event_id"])]["task_id_hash"]
        for private in private_rows
        if private["decision"] == "reverse_preference"
    }
    max_task_events = max(tasks.values(), default=0)
    max_task_share = max_task_events / max(1, len(public_rows))
    checks = {
        "enough_events": len(public_rows) >= THRESHOLDS["min_events"],
        "enough_reverse_events": decisions.get("reverse_preference", 0)
        >= THRESHOLDS["min_reverse_events"],
        "enough_reverse_tasks": len(reverse_tasks) >= THRESHOLDS["min_reverse_tasks"],
        "task_event_cap": max_task_events <= THRESHOLDS["max_events_per_task"],
        "not_task_dominated": max_task_share <= THRESHOLDS["max_task_event_share"],
        "visible_prefixes_exported": all(
            "reference_free_prefix_steps" in row for row in public_rows
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
        "stage": "toolsandbox_v43_multi_prefix_training_data",
        "passed": passed,
        "events": len(public_rows),
        "tasks": len(tasks),
        "reverse_tasks": len(reverse_tasks),
        "decisions": dict(sorted(decisions.items())),
        "modes": dict(Counter(row["mode"] for row in public_rows)),
        "max_task_events": max_task_events,
        "max_task_event_share": max_task_share,
        "event_set_hash": event_set_hash(public_rows),
        "source_signal_sha256": file_sha256(signal_path),
        "source_summary_sha256": file_sha256(summary_path),
        "source_gate_sha256": file_sha256(gate_path),
        "source_protocol_sha256": summary["protocol_lock_sha256"],
        "public_sha256": file_sha256(public_path),
        "private_sha256": file_sha256(private_path),
        "train_sha256": file_sha256(train_path),
        "checks": checks,
        "thresholds": THRESHOLDS,
        "model_inputs": ["prompt", "candidate action completion"],
        "official_branch_metrics_in_training_file": False,
        "protected_outcomes_in_prompt": False,
        "branch_receipts_exported": False,
        "reference_actions_read_or_exported": False,
        "scope": "frozen multi-prefix ToolSandbox preference training data",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(
        args.output_dir / "data_gate.json",
        {
            "passed": passed,
            "stage": "toolsandbox_v43_multi_prefix_data_gate",
            "checks": checks,
            "thresholds": THRESHOLDS,
            "events": len(public_rows),
            "reverse_events": decisions.get("reverse_preference", 0),
            "reverse_tasks": len(reverse_tasks),
            "next_step": (
                "train the frozen V4.3 comparison"
                if passed
                else "stop before training; causal diversity is still insufficient"
            ),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
