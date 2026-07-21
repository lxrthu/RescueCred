#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rescuecredit.compensation_trap import (
    LABELS,
    PRIVATE_FIELDS,
    PUBLIC_FIELDS,
    deterministic_split,
    validate_benchmark_package_data,
)
from rescuecredit.frozen_bank import file_sha256, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.benchmark_dir
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    public = read_jsonl(root / "public_events.jsonl")
    private = read_jsonl(root / "private_outcomes.jsonl")
    splits = read_jsonl(root / "splits.jsonl")
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    card = json.loads((root / "dataset_card.json").read_text(encoding="utf-8"))
    public_by_id = {str(row["event_id"]): row for row in public}
    private_by_id = {str(row["event_id"]): row for row in private}
    split_by_id = {str(row["event_id"]): row for row in splits}
    validation = validate_benchmark_package_data(
        public, private, splits, schema, manifest
    )
    if not all(validation.values()):
        raise ValueError({"benchmark_validation_failed": validation})
    if not (len(public_by_id) == len(public) == len(private_by_id) == len(private)):
        raise ValueError("benchmark contains duplicate event IDs")
    if set(public_by_id) != set(private_by_id) or set(public_by_id) != set(split_by_id):
        raise ValueError("benchmark public/private/split event sets differ")
    if any(set(row) != set(PUBLIC_FIELDS) for row in public):
        raise ValueError("public benchmark field boundary changed")
    if any(set(row) != set(PRIVATE_FIELDS) for row in private):
        raise ValueError("private benchmark field boundary changed")
    if any(row["decision"] not in LABELS for row in private):
        raise ValueError("benchmark contains unsupported labels")
    if any(
        str(row["task_id_hash"])
        != hashlib.sha256(str(row["scenario_name"]).encode("utf-8")).hexdigest()
        for row in public
    ):
        raise ValueError("benchmark scenario/task identity mismatch")
    task_split = {}
    for row in splits:
        task = str(row["task_id_hash"])
        split = str(row["split"])
        if task in task_split and task_split[task] != split:
            raise ValueError("task leakage across benchmark splits")
        if split != deterministic_split(task):
            raise ValueError("benchmark split differs from deterministic rule")
        task_split[task] = split
    hashes = {
        "public_sha256": file_sha256(root / "public_events.jsonl"),
        "private_sha256": file_sha256(root / "private_outcomes.jsonl"),
        "splits_sha256": file_sha256(root / "splits.jsonl"),
        "schema_sha256": file_sha256(root / "schema.json"),
        "dataset_card_sha256": file_sha256(root / "dataset_card.json"),
    }
    if any(manifest.get(key) != value for key, value in hashes.items()):
        raise ValueError("benchmark manifest hash mismatch")
    if schema.get("version") != "compensation_trap_benchmark_v1" or schema.get(
        "public_fields"
    ) != list(PUBLIC_FIELDS) or schema.get("private_fields") != list(PRIVATE_FIELDS):
        raise ValueError("benchmark schema changed")
    if card.get("release_authorized") is not False or card.get(
        "requires_upstream_license_review"
    ) is not True or set(card.get("provenance_only_fields", [])) != {
        "scenario_name",
        "reference_free_prefix_steps",
        "source",
        "task_id_hash",
    }:
        raise ValueError("benchmark release/license boundary changed")
    split_event_counts = dict(
        sorted(Counter(str(row["split"]) for row in splits).items())
    )
    split_task_counts = {
        split: len({task for task, assigned in task_split.items() if assigned == split})
        for split in ("train", "development", "test")
    }
    label_counts = dict(sorted(Counter(str(row["decision"]) for row in private).items()))
    source_names = [str(row["name"]) for row in manifest.get("sources", [])]
    if len(source_names) != len(set(source_names)) or not source_names:
        raise ValueError("benchmark source inventory is invalid")
    if not {str(row["source"]) for row in public} <= set(source_names):
        raise ValueError("benchmark row source is not manifest-bound")
    if not (
        manifest.get("events") == len(public)
        and manifest.get("tasks") == len(task_split)
        and manifest.get("label_counts") == label_counts
        and manifest.get("split_event_counts") == split_event_counts
        and manifest.get("split_task_counts") == split_task_counts
    ):
        raise ValueError("benchmark manifest statistics mismatch")
    print(
        json.dumps(
            {
                "passed": True,
                "events": len(public),
                "tasks": len(task_split),
                "checks": {
                    **validation,
                    "hashes": True,
                    "public_private_boundary": True,
                    "task_disjoint_splits": True,
                    "labels": True,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
