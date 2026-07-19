#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

from environments.toolsandbox import ToolSandboxRuntime, canonical_action
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_credit import validate_branch_credit_evidence
from rescuecredit.toolsandbox_preference import (
    NONZERO_DECISIONS,
    event_set_hash,
    public_preference_prompt,
)


def _function(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    function = entry.get("function", entry)
    return function if isinstance(function, Mapping) else {}


def _relevant_schemas(
    schemas: list[dict[str, Any]], action_a: Mapping[str, Any], action_b: Mapping[str, Any]
) -> list[dict[str, Any]]:
    names = {str(action_a["tool"]), str(action_b["tool"])}
    selected = [entry for entry in schemas if str(_function(entry).get("name")) in names]
    if {str(_function(entry).get("name")) for entry in selected} != names:
        raise ValueError("candidate action tool is absent from the public schema catalog")
    return selected


def _branch_metrics(branch: Mapping[str, Any], horizon: int) -> Dict[str, Any]:
    validated = validate_branch_credit_evidence(branch, horizon=horizon)
    score = branch["score"]
    return {
        "terminal_similarity": validated["final_similarity"],
        "progress_auc": validated["progress_auc"],
        "tool_errors": int(branch["tool_errors"]),
        "official_turn_count": int(score["turn_count"]),
        "branch_steps": int(branch["steps"]),
        "score_source": str(score["source"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-events", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--quality-gate", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "evaluation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    def load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    summary = load(args.audit_summary)
    gate = load(args.quality_gate)
    if summary.get("status") != "completed":
        raise RuntimeError("source ToolSandbox audit did not complete")
    if summary.get("credit_mode") != "lexicographic_v4":
        raise RuntimeError("source audit does not use V4 lexicographic credit")
    if summary.get("harness_interface") != "tool_id_v2":
        raise RuntimeError("source audit does not use the frozen Tool-ID interface")
    if summary.get("protocol_validated") is not True:
        raise RuntimeError("source audit protocol was not validated")
    if summary.get("event_file_sha256") != file_sha256(args.signal_events):
        raise RuntimeError("signal event file does not match its audit summary")
    if args.role == "train" and gate.get("passed") is not True:
        raise RuntimeError("training source requires the passed V4.1 fresh gate")
    if args.role == "evaluation" and gate.get("mechanism_passed") is not True:
        raise RuntimeError("evaluation source requires a passed fresh mechanism gate")

    horizon = int(summary["horizon"])
    source_rows = [
        row
        for row in read_jsonl(args.signal_events)
        if row.get("replay_valid") is True
        and row.get("decision") in NONZERO_DECISIONS
        and row.get("mode")
        in {"controlled_missing_argument", "natural_visible_error_repair"}
    ]
    runtime = ToolSandboxRuntime()
    scenarios = runtime.scenarios()
    public_rows = []
    private_rows = []
    train_rows = []
    for row in source_rows:
        scenario_name = str(row["scenario_name"])
        if scenario_name not in scenarios:
            raise ValueError(f"unknown frozen ToolSandbox scenario: {scenario_name}")
        context = runtime.prepare(scenarios[scenario_name])
        action_a = canonical_action(row["action_a"])
        action_b = canonical_action(row["action_b"])
        schemas = _relevant_schemas(runtime.tool_schemas(context), action_a, action_b)
        prompt = public_preference_prompt(
            visible_history=runtime.visible_history(context),
            public_tool_schemas=schemas,
            action_a=action_a,
            action_b=action_b,
        )
        public = {
            "event_id": str(row["event_id"]),
            "task_id_hash": str(row["task_id_hash"]),
            "mode": str(row["mode"]),
            "prompt": prompt,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_path = args.output_dir / "events.public.jsonl"
    private_path = args.output_dir / "outcomes.private.jsonl"
    write_jsonl(public_path, public_rows)
    write_jsonl(private_path, private_rows)
    train_path = args.output_dir / "train.jsonl"
    if args.role == "train":
        write_jsonl(train_path, train_rows)

    manifest = {
        "status": "frozen",
        "stage": "toolsandbox_v41_preference_data",
        "role": args.role,
        "events": len(public_rows),
        "event_set_hash": event_set_hash(public_rows),
        "decisions": dict(Counter(row["decision"] for row in private_rows)),
        "modes": dict(Counter(row["mode"] for row in public_rows)),
        "decision_bases": dict(
            Counter(row["decision_basis"] for row in private_rows)
        ),
        "horizon": horizon,
        "source_signal_sha256": file_sha256(args.signal_events),
        "source_summary_sha256": file_sha256(args.audit_summary),
        "source_gate_sha256": file_sha256(args.quality_gate),
        "public_sha256": file_sha256(public_path),
        "private_sha256": file_sha256(private_path),
        "train_sha256": file_sha256(train_path) if train_path.is_file() else None,
        "model_inputs": ["prompt", "candidate action completion"],
        "training_label_fields": ["decision", "decision_basis", "causal_weight"],
        "official_branch_metrics_in_training_file": False,
        "protected_outcomes_in_prompt": False,
        "branch_receipts_exported": False,
        "reference_actions_read_or_exported": False,
        "scope": (
            "frozen V4.1 preference training data"
            if args.role == "train"
            else "fresh held-out preference evaluation data"
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
