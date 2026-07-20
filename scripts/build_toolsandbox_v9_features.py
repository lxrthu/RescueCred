#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_active_shadow_v9 import build_two_step_features
from rescuecredit.toolsandbox_selective_router import reverse_target
from scripts.freeze_toolsandbox_v9_protocol import CONFIG, PROTOCOL_STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-events", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--v5-feature-cache", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch

    started = time.time()
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("config") != CONFIG:
        raise ValueError("invalid V9 protocol")
    identities = {
        args.raw_events: protocol["raw_events_sha256"],
        args.train_file: protocol["train_file_sha256"],
        args.v5_feature_cache: protocol["v5_feature_cache_sha256"],
    }
    if any(file_sha256(path) != digest for path, digest in identities.items()):
        raise ValueError("V9 input identity mismatch")

    raw_rows = read_jsonl(args.raw_events)
    raw_by_id = {str(row["event_id"]): row for row in raw_rows}
    train_rows = read_jsonl(args.train_file)
    train_ids = [str(row["event_id"]) for row in train_rows]
    if len(raw_by_id) != len(raw_rows) or len(train_ids) != len(set(train_ids)):
        raise ValueError("V9 raw/train event identifiers are not unique")
    v5 = torch.load(args.v5_feature_cache, map_location="cpu", weights_only=True)
    v5_by_id = {
        str(event_id): index for index, event_id in enumerate(v5["event_ids"])
    }
    if len(v5_by_id) != len(v5["event_ids"]):
        raise ValueError("V9 V5 event identifiers are not unique")
    active_features = []
    static_features = []
    labels = []
    event_ids = []
    task_ids = []
    second_receipt_a = 0
    second_receipt_b = 0
    for train_row in train_rows:
        event_id = str(train_row["event_id"])
        raw = raw_by_id.get(event_id)
        if raw is None or event_id not in v5_by_id:
            raise ValueError(f"V9 event missing from raw/static cache: {event_id}")
        if raw.get("decision") != train_row.get("decision"):
            raise ValueError("V9 raw/train decision mismatch")
        if str(raw.get("task_id_hash")) != str(train_row.get("task_id_hash")):
            raise ValueError("V9 raw/train task identity mismatch")
        active_features.append(
            build_two_step_features(raw, hash_dimension=CONFIG["hash_dimension"])
        )
        static_features.append(v5["semantic_features"][v5_by_id[event_id]].float())
        labels.append(reverse_target(str(train_row["decision"])))
        event_ids.append(event_id)
        task_ids.append(str(train_row["task_id_hash"]))
        second_receipt_a += int(len(raw["branch_a"]["receipts"]) >= 2)
        second_receipt_b += int(len(raw["branch_b"]["receipts"]) >= 2)

    payload = {
        "active_features": torch.tensor(active_features, dtype=torch.float32),
        "static_features": torch.stack(static_features),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "event_ids": event_ids,
        "task_ids": task_ids,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "active_shadow_features.pt"
    torch.save(payload, cache_path)
    manifest = {
        "status": "completed",
        "stage": "toolsandbox_v9_two_step_feature_cache",
        "events": len(labels),
        "tasks": len(set(task_ids)),
        "reverse_events": sum(labels),
        "rescue_events": len(labels) - sum(labels),
        "second_receipt_a_events": second_receipt_a,
        "second_receipt_b_events": second_receipt_b,
        "active_feature_dim": int(payload["active_features"].shape[1]),
        "static_feature_dim": int(payload["static_features"].shape[1]),
        "feature_cache_sha256": file_sha256(cache_path),
        "raw_events_sha256": file_sha256(args.raw_events),
        "train_file_sha256": file_sha256(args.train_file),
        "v5_feature_cache_sha256": file_sha256(args.v5_feature_cache),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "protected_fields_used": [],
        "receipt_horizon": 2,
        "third_or_later_receipts_used": False,
        "continuation_policy_calls_per_probed_event": 2,
        "maximum_tool_executions_per_probed_event": 4,
        "dataset_mean_tool_executions_per_probed_event": (
            2.0 + (second_receipt_a + second_receipt_b) / len(labels)
        ),
        "full_trajectory_fields_used": False,
        "official_evaluator_features_used": False,
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "feature_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
