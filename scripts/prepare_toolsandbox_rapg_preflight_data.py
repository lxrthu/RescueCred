#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from environments.toolsandbox import action_schema_complete, canonical_action
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import public_preference_prompt
from scripts.freeze_toolsandbox_v44_candidate_protocol import PROTOCOL_STATUS
from scripts.prepare_toolsandbox_v41_preference_data import _relevant_schemas


def _terminal_similarity(branch: Mapping[str, Any]) -> float:
    score = branch.get("score")
    if not isinstance(score, Mapping):
        raise ValueError("replay-valid branch lacks official score")
    return float(score["similarity"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-events", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite RAPG source split")
    summary = json.loads(args.audit_summary.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("role") != "full":
        raise ValueError("RAPG preflight requires frozen V4.4 full protocol")
    if summary.get("status") != "completed" or summary.get("role") != "full":
        raise ValueError("RAPG preflight requires completed V4.4 full audit")
    if summary.get("protocol_lock_sha256") != file_sha256(args.protocol_lock):
        raise ValueError("V4.4 audit/protocol identity mismatch")
    if summary.get("event_file_sha256") != file_sha256(args.raw_events):
        raise ValueError("V4.4 raw event identity mismatch")
    if summary.get("snapshot_audit", {}).get("exact") is not True:
        raise ValueError("V4.4 snapshot replay is not exact")

    rows = [
        row
        for row in read_jsonl(args.raw_events)
        if row.get("mode") == "both_valid_candidate_pair"
        and row.get("replay_valid") is True
    ]
    if len(rows) < 100:
        raise ValueError("RAPG preflight requires at least 100 replay-valid pairs")
    public_rows: list[dict[str, Any]] = []
    executed_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    for row in rows:
        history = row.get("treatment_visible_history")
        schemas = row.get("treatment_public_tool_schemas")
        if not isinstance(history, list) or not isinstance(schemas, list):
            raise ValueError("RAPG source lacks public treatment context")
        action_a = canonical_action(row["action_a"])
        action_b = canonical_action(row["action_b"])
        if action_a == action_b:
            raise ValueError("RAPG candidate pair contains identical actions")
        if not action_schema_complete(action_a, schemas) or not action_schema_complete(
            action_b, schemas
        ):
            raise ValueError("RAPG candidate action is not public-schema complete")
        relevant = _relevant_schemas(schemas, action_a, action_b)
        event_id = str(row["event_id"])
        public_rows.append(
            {
                "event_id": event_id,
                "task_id_hash": str(row["task_id_hash"]),
                "prompt": public_preference_prompt(
                    visible_history=history,
                    public_tool_schemas=relevant,
                    action_a=action_a,
                    action_b=action_b,
                ),
                "action_a": action_a,
                "action_b": action_b,
            }
        )
        executed_rows.append(
            {
                "event_id": event_id,
                "executed_b_return": _terminal_similarity(row["branch_b"]),
            }
        )
        shadow_rows.append(
            {
                "event_id": event_id,
                "shadow_a_return": _terminal_similarity(row["branch_a"]),
            }
        )
    for collection in (public_rows, executed_rows, shadow_rows):
        collection.sort(key=lambda row: row["event_id"])
    event_ids = [row["event_id"] for row in public_rows]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("RAPG source event ids are not unique")

    args.output_dir.mkdir(parents=True)
    public_path = args.output_dir / "events.public.jsonl"
    executed_path = args.output_dir / "executed_b_returns.jsonl"
    shadow_path = args.output_dir / "shadow_a_returns.private.jsonl"
    write_jsonl(public_path, public_rows)
    write_jsonl(executed_path, executed_rows)
    write_jsonl(shadow_path, shadow_rows)
    manifest = {
        "status": "completed",
        "stage": "toolsandbox_rapg_surrogate_preflight_source",
        "events": len(public_rows),
        "tasks": len({row["task_id_hash"] for row in public_rows}),
        "selection": "all replay-valid both-valid V4.4 pairs; no delta/direction filter",
        "outcome_direction_filter_used": False,
        "replay_validity_conditioned": True,
        "deployment_stream_representative": False,
        "role": "development_surrogate_preflight_only",
        "public_sha256": file_sha256(public_path),
        "executed_b_sha256": file_sha256(executed_path),
        "shadow_a_sha256": file_sha256(shadow_path),
        "raw_events_sha256": file_sha256(args.raw_events),
        "audit_summary_sha256": file_sha256(args.audit_summary),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "ground_truth": "official paired ToolSandbox terminal_similarity",
        "shadow_source_cost_events": len(shadow_rows),
        "claim_boundary": (
            "offline outcome-conditioned replay-valid surrogate preflight; cannot authorize a policy claim"
        ),
    }
    write_json(args.output_dir / "source_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
