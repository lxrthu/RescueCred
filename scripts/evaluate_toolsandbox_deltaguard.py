#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rescuecredit.deltaguard_baseline import compute_v7_baseline_scores
from rescuecredit.deltaguard_evaluation import evaluate_deltaguard, label_from_decision
from rescuecredit.deltaguard_probe import action_hash
from rescuecredit.deltaguard_protocol import (
    PROTOCOL_STATUS,
    export_public_event,
    load_public_sources,
    verify_protocol_source_identity,
)
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--label-events", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    protocol = _load(args.protocol_lock)
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("invalid DeltaGuard protocol")
    verify_protocol_source_identity(protocol)
    checkpoint = Path(protocol["v7_checkpoint"])
    if file_sha256(checkpoint) != protocol["v7_checkpoint_sha256"]:
        raise ValueError("V7 baseline checkpoint drift")
    source_path = args.collection_dir / "source_ledger.jsonl"
    probe_path = args.collection_dir / "probe_ledger.jsonl"
    manifest_path = args.collection_dir / "collection_manifest.json"
    manifest = _load(manifest_path)
    if manifest.get("protocol_lock_sha256") != file_sha256(args.protocol_lock):
        raise ValueError("collection/protocol mismatch")
    if manifest.get("source_ledger_sha256") != file_sha256(source_path):
        raise ValueError("source ledger drift")
    if manifest.get("probe_ledger_sha256") != file_sha256(probe_path):
        raise ValueError("probe ledger drift")
    source_rows = read_jsonl(source_path)
    probe_rows = read_jsonl(probe_path)
    public_by_id = {
        str(row["event_id"]): row
        for row in load_public_sources(
            [Path(item["path"]) for item in protocol["public_sources"]]
        )
    }
    label_rows = []
    for path in args.label_events:
        label_rows.extend(read_jsonl(path))
    public_manifest = _load(Path(protocol["public_bank_manifest"]))
    label_hashes = [file_sha256(path) for path in args.label_events]
    if label_hashes != public_manifest.get("raw_source_sha256"):
        raise ValueError("label sources do not match the sealed public-bank provenance")
    raw_by_id = {str(row["event_id"]): row for row in label_rows}
    if len(raw_by_id) != len(label_rows):
        raise ValueError("label banks contain duplicate event IDs")
    frozen_by_id = {str(row["event_id"]): row for row in protocol["source_events"]}
    for event_id in frozen_by_id:
        raw = raw_by_id.get(event_id)
        public = public_by_id.get(event_id)
        if raw is None or public is None:
            raise ValueError(f"label/public identity missing for {event_id}")
        if public != export_public_event(raw):
            raise ValueError(f"sealed public projection mismatch for {event_id}")
        if raw.get("replay_valid") is not True:
            raise ValueError("label source event is not exact-replay valid")
        if action_hash(raw["action_a"]) != frozen_by_id[event_id]["action_hash_a"]:
            raise ValueError("label bank A action identity mismatch")
        if action_hash(raw["action_b"]) != frozen_by_id[event_id]["action_hash_b"]:
            raise ValueError("label bank B action identity mismatch")
    labels = {
        str(row["event_id"]): label_from_decision(str(raw_by_id[str(row["event_id"])]["decision"]))
        for row in source_rows
    }
    oof_path = Path(protocol["v7_oof"]) if protocol.get("v7_oof") else None
    if oof_path is not None and file_sha256(oof_path) != protocol["v7_oof_sha256"]:
        raise ValueError("V7 OOF identity drift")
    baselines, baseline_sources = compute_v7_baseline_scores(
        probe_rows=probe_rows,
        checkpoint_path=checkpoint,
        hash_dimension=int(protocol["v7_hash_dimension"]),
        oof_path=oof_path,
    )
    config = protocol["config"]
    summary = evaluate_deltaguard(
        source_rows=source_rows,
        probe_rows=probe_rows,
        labels=labels,
        baseline_scores=baselines,
        min_class_per_family=int(config["min_class_per_family"]),
        min_auc=float(config["min_typed_delta_roc_auc"]),
        min_auc_gain=float(config["min_auc_gain_over_v7"]),
        max_probe_rate=float(config["max_probe_rate"]),
        alpha=float(config["risk_alpha"]),
    )
    predictions = [
        {
            "event_id": str(row["event_id"]),
            "task_id_hash": str(row["task_id_hash"]),
            "family": str(row["family"]),
            "label": labels[str(row["event_id"])],
            "selected": bool(row["selected"]),
            "typed_delta_score": float(
                next(
                    (probe["reverse_score"] for probe in probe_rows if probe["event_id"] == row["event_id"]),
                    0.5,
                )
            ),
            "contract_score": float(
                next(
                    (probe["contract_reverse_score"] for probe in probe_rows if probe["event_id"] == row["event_id"]),
                    0.5,
                )
            ),
            "v7_score": baselines.get(str(row["event_id"])),
            "v7_score_source": baseline_sources.get(str(row["event_id"])),
        }
        for row in source_rows
    ]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    summary.update(
        {
            "stage": "toolsandbox_deltaguard_evaluation",
            "protocol_lock_sha256": file_sha256(args.protocol_lock),
            "collection_manifest_sha256": file_sha256(manifest_path),
            "source_ledger_sha256": file_sha256(source_path),
            "probe_ledger_sha256": file_sha256(probe_path),
            "predictions_sha256": file_sha256(prediction_path),
            "labels_revealed_after_collection": True,
            "ground_truth_source": "exact Shadow decision from frozen ToolSandbox paired branches",
            "label_source_sha256": label_hashes,
            "v7_baseline_score_sources": {
                source: sum(value == source for value in baseline_sources.values())
                for source in sorted(set(baseline_sources.values()))
            },
            "wall_time_sec": time.time() - started,
        }
    )
    write_json(output / "evaluation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
