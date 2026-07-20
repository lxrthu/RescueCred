#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_active_shadow_v8 import build_v8_features
from rescuecredit.toolsandbox_selective_router import reverse_target
from scripts.freeze_toolsandbox_v8_protocol import CONFIG, PROTOCOL_STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-events", type=Path, required=True)
    parser.add_argument("--collection-summary", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--v5-feature-cache", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch

    started = time.time()
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    collection = json.loads(args.collection_summary.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("config") != CONFIG:
        raise ValueError("invalid V8 feature protocol")
    if collection.get("event_file_sha256") != file_sha256(args.state_events):
        raise ValueError("V8 state event identity mismatch")
    if collection.get("protocol_lock_sha256") != file_sha256(args.protocol_lock):
        raise ValueError("V8 collection protocol mismatch")
    if collection.get("official_evaluator_called") is not False:
        raise ValueError("V8 collection called the official evaluator")
    if file_sha256(args.train_file) != protocol.get("train_file_sha256"):
        raise ValueError("V8 train identity mismatch")
    if file_sha256(args.v5_feature_cache) != protocol.get("v5_feature_cache_sha256"):
        raise ValueError("V8 static cache identity mismatch")

    state_rows = read_jsonl(args.state_events)
    state_by_id = {str(row["event_id"]): row for row in state_rows}
    train_rows = read_jsonl(args.train_file)
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
        state = state_by_id.get(event_id)
        if state is None or event_id not in v5_by_id:
            raise ValueError(f"V8 event missing from state/static cache: {event_id}")
        active_features.append(
            build_v8_features(state, hash_dimension=CONFIG["hash_dimension"])
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
        "stage": "toolsandbox_v8_visible_state_feature_cache",
        "events": len(labels),
        "tasks": len(set(task_ids)),
        "reverse_events": sum(labels),
        "rescue_events": len(labels) - sum(labels),
        "active_feature_dim": int(payload["active_features"].shape[1]),
        "static_feature_dim": int(payload["static_features"].shape[1]),
        "feature_cache_sha256": file_sha256(cache_path),
        "state_events_sha256": file_sha256(args.state_events),
        "collection_summary_sha256": file_sha256(args.collection_summary),
        "train_file_sha256": file_sha256(args.train_file),
        "v5_feature_cache_sha256": file_sha256(args.v5_feature_cache),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "protected_fields_used": [],
        "official_evaluator_features_used": False,
        "hidden_context_features_used": False,
        "full_trajectory_fields_used": False,
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "feature_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
