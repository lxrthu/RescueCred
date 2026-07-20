#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_active_shadow import build_active_shadow_features
from rescuecredit.toolsandbox_selective_router import reverse_target
from scripts.freeze_toolsandbox_v7_protocol import CONFIG, PROTOCOL_STATUS


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
        raise ValueError("invalid V7 ActiveShadow protocol")
    identities = {
        args.raw_events: protocol["raw_events_sha256"],
        args.train_file: protocol["train_file_sha256"],
        args.v5_feature_cache: protocol["v5_feature_cache_sha256"],
    }
    if any(file_sha256(path) != digest for path, digest in identities.items()):
        raise ValueError("V7 input identity mismatch")

    raw_rows = read_jsonl(args.raw_events)
    train_rows = read_jsonl(args.train_file)
    raw_by_id = {str(row["event_id"]): row for row in raw_rows}
    if len(raw_by_id) != len(raw_rows):
        raise ValueError("raw V4.4 event identifiers are not unique")
    v5 = torch.load(args.v5_feature_cache, map_location="cpu", weights_only=True)
    v5_by_id = {
        str(event_id): index for index, event_id in enumerate(v5["event_ids"])
    }

    active_features = []
    static_features = []
    labels = []
    event_ids = []
    task_ids = []
    for train_row in train_rows:
        event_id = str(train_row["event_id"])
        if event_id not in raw_by_id or event_id not in v5_by_id:
            raise ValueError(f"V7 event missing from raw or V5 cache: {event_id}")
        raw = raw_by_id[event_id]
        if raw.get("decision") != train_row.get("decision"):
            raise ValueError("V7 raw/train decision mismatch")
        active_features.append(
            build_active_shadow_features(raw, hash_dimension=CONFIG["hash_dimension"])
        )
        static_features.append(v5["semantic_features"][v5_by_id[event_id]].float())
        labels.append(reverse_target(str(train_row["decision"])))
        event_ids.append(event_id)
        task_ids.append(str(train_row["task_id_hash"]))

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
        "stage": "toolsandbox_v7_active_shadow_feature_cache",
        "events": len(labels),
        "tasks": len(set(task_ids)),
        "reverse_events": sum(labels),
        "rescue_events": len(labels) - sum(labels),
        "active_feature_dim": int(payload["active_features"].shape[1]),
        "static_feature_dim": int(payload["static_features"].shape[1]),
        "feature_cache_sha256": file_sha256(cache_path),
        "raw_events_sha256": file_sha256(args.raw_events),
        "train_file_sha256": file_sha256(args.train_file),
        "v5_feature_cache_sha256": file_sha256(args.v5_feature_cache),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "visible_transition_inputs": [
            "action A",
            "action B",
            "first receipt A content and exception",
            "first receipt B content and exception",
        ],
        "protected_fields_used": [],
        "raw_receipt_text_exported": False,
        "hidden_state_diff_available": False,
        "full_trajectory_fields_used": False,
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "feature_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
