#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from rescuecredit.compensation_trap import (
    LABELS,
    PRIVATE_FIELDS,
    PUBLIC_FIELDS,
    deterministic_split,
    private_projection,
    public_projection,
)
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_credit import (
    lexicographic_counterfactual_regret,
    validate_branch_credit_evidence,
)


def validated_official_credit(raw: dict, *, horizon: int, atol: float) -> dict:
    if raw.get("credit_mode") != "lexicographic_v4":
        raise ValueError("benchmark source credit mode is not lexicographic_v4")
    validate_branch_credit_evidence(raw["branch_a"], horizon=horizon, atol=atol)
    validate_branch_credit_evidence(raw["branch_b"], horizon=horizon, atol=atol)
    recomputed = lexicographic_counterfactual_regret(
        raw["branch_a"], raw["branch_b"], horizon=horizon, atol=atol
    )
    stored_matches = (
        recomputed["decision"] == raw.get("decision")
        and recomputed["decision_basis"] == raw.get("decision_basis")
        and math.isclose(
            float(recomputed["decision_value"]),
            float(raw.get("decision_value")),
            rel_tol=1e-10,
            abs_tol=atol,
        )
        and math.isclose(
            float(recomputed["causal_weight"]),
            float(raw.get("causal_weight")),
            rel_tol=1e-10,
            abs_tol=atol,
        )
        and recomputed["components"] == raw.get("credit_components")
    )
    if not stored_matches:
        raise ValueError(
            f"stored credit differs from official recomputation: {raw.get('event_id')}"
        )
    return recomputed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-events", type=Path, action="append", required=True)
    parser.add_argument("--source-name", action="append", required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.raw_events) != len(args.source_name):
        raise ValueError("each raw event file requires one source name")
    if len(set(args.source_name)) != len(args.source_name):
        raise ValueError("compensation benchmark source names must be unique")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite compensation benchmark")
    public_rows = []
    private_rows = []
    event_ids: set[str] = set()
    sources = []
    for path, source in zip(args.raw_events, args.source_name, strict=True):
        sources.append(
            {
                "name": str(source),
                "file_name": path.name,
                "sha256": file_sha256(path),
            }
        )
        for raw in read_jsonl(path):
            if raw.get("mode") != "both_valid_candidate_pair" or raw.get(
                "replay_valid"
            ) is not True:
                continue
            recomputed = validated_official_credit(
                raw, horizon=args.horizon, atol=args.atol
            )
            if recomputed["decision"] not in LABELS:
                continue
            event_id = str(raw.get("event_id", ""))
            compound_id = f"{source}:{event_id}"
            if not event_id or compound_id in event_ids:
                raise ValueError(f"missing or duplicate source event ID: {compound_id}")
            expected_task_hash = hashlib.sha256(
                str(raw.get("scenario_name", "")).encode("utf-8")
            ).hexdigest()
            if str(raw.get("task_id_hash", "")) != expected_task_hash:
                raise ValueError(f"scenario/task identity mismatch: {compound_id}")
            event_ids.add(compound_id)
            public = public_projection(raw, str(source))
            private = private_projection(raw, str(source))
            public["event_id"] = compound_id
            private["event_id"] = compound_id
            public_rows.append(public)
            private_rows.append(private)
    if not public_rows:
        raise ValueError("no replay-valid nonzero compensation events")
    public_rows.sort(key=lambda row: row["event_id"])
    private_rows.sort(key=lambda row: row["event_id"])
    split_rows = [
        {
            "event_id": row["event_id"],
            "task_id_hash": row["task_id_hash"],
            "split": deterministic_split(str(row["task_id_hash"])),
        }
        for row in public_rows
    ]
    args.output_dir.mkdir(parents=True)
    public_path = args.output_dir / "public_events.jsonl"
    private_path = args.output_dir / "private_outcomes.jsonl"
    split_path = args.output_dir / "splits.jsonl"
    schema_path = args.output_dir / "schema.json"
    card_path = args.output_dir / "dataset_card.json"
    write_jsonl(public_path, public_rows)
    write_jsonl(private_path, private_rows)
    write_jsonl(split_path, split_rows)
    write_json(
        schema_path,
        {
            "version": "compensation_trap_benchmark_v1",
            "public_fields": list(PUBLIC_FIELDS),
            "private_fields": list(PRIVATE_FIELDS),
            "labels": list(LABELS),
            "split_rule": "sha256(task_id_hash) prefix modulo 10: 0-5 train, 6-7 development, 8-9 test",
            "claim_boundary": "historical replay-valid nonzero events; task-disjoint benchmark construction, not untouched confirmation",
        },
    )
    write_json(
        card_path,
        {
            "name": "Compensation Trap Exact Shadow Benchmark",
            "version": "v1-development",
            "intended_use": "task-disjoint diagnosis of Rescue versus Reverse Harness interventions",
            "ground_truth": "official paired ToolSandbox evaluator evidence with lexicographic tie-breaking",
            "public_private_boundary": "public events contain deployment-visible history, schemas, and A/B actions; branch outcomes and direction labels remain private",
            "provenance_only_fields": [
                "scenario_name",
                "reference_free_prefix_steps",
                "source",
                "task_id_hash",
            ],
            "deployable_baseline_feature_rule": "baselines may use only treatment_visible_history, treatment_public_tool_schemas, action_a, and action_b; provenance-only fields are forbidden",
            "limitations": [
                "historical replay-valid nonzero events only",
                "not deployment-stream representative",
                "collision claims are conditional on frozen representations",
            ],
            "requires_upstream_license_review": True,
            "release_authorized": False,
        },
    )
    public_by_id = {row["event_id"]: row for row in public_rows}
    split_by_id = {row["event_id"]: row["split"] for row in split_rows}
    label_counts = Counter(row["decision"] for row in private_rows)
    manifest = {
        "status": "completed",
        "version": "compensation_trap_benchmark_v1",
        "events": len(public_rows),
        "tasks": len({row["task_id_hash"] for row in public_rows}),
        "label_counts": dict(sorted(label_counts.items())),
        "split_event_counts": dict(sorted(Counter(split_by_id.values()).items())),
        "split_task_counts": {
            split: len(
                {
                    public_by_id[event_id]["task_id_hash"]
                    for event_id, assigned in split_by_id.items()
                    if assigned == split
                }
            )
            for split in ("train", "development", "test")
        },
        "sources": sources,
        "public_sha256": file_sha256(public_path),
        "private_sha256": file_sha256(private_path),
        "splits_sha256": file_sha256(split_path),
        "schema_sha256": file_sha256(schema_path),
        "dataset_card_sha256": file_sha256(card_path),
        "outcome_direction_filter_used": True,
        "official_branch_evidence_recomputed": True,
        "label_rule": "lexicographic_counterfactual_regret_v4",
        "credit_mode": "lexicographic_v4",
        "horizon": args.horizon,
        "atol": args.atol,
        "selection": "all replay-valid nonzero candidate pairs in the explicitly supplied historical sources",
        "paper_role": "development benchmark and collision audit only",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
