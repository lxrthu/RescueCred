#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_selective_router import roc_auc
from scripts.freeze_toolsandbox_v7_protocol import CONFIG, GATE as V7_GATE


PROTOCOL_STATUS = "frozen_before_toolsandbox_v9_two_step_offline_probe"
GATE = {**V7_GATE, "require_two_step_auc_above_one_step": True}
SOURCE_PATHS = (
    "rescuecredit/toolsandbox_active_shadow.py",
    "rescuecredit/toolsandbox_active_shadow_v9.py",
    "rescuecredit/toolsandbox_selective_router.py",
    "scripts/freeze_toolsandbox_v9_protocol.py",
    "scripts/build_toolsandbox_v9_features.py",
    "scripts/train_toolsandbox_v9_active_shadow.py",
    "scripts/train_toolsandbox_v7_active_shadow.py",
    "scripts/check_toolsandbox_v9_gate.py",
    "scripts/check_toolsandbox_v7_gate.py",
    "scripts/cloud/run_toolsandbox_v9_two_step_seed42.sh",
    "refine-logs/TOOLSANDBOX_V9_TWO_STEP_PLAN.md",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v44-root", type=Path, required=True)
    parser.add_argument("--v5-root", type=Path, required=True)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    raw = args.v44_root / "full_offset85_h8/candidate_events.jsonl"
    raw_summary_path = args.v44_root / "full_offset85_h8/audit_summary.json"
    train = args.v44_root / "data/train.jsonl"
    train_manifest_path = args.v44_root / "data/manifest.json"
    v5_cache = args.v5_root / "features/train_features.pt"
    v5_manifest_path = args.v5_root / "features/feature_manifest.json"
    v7_gate_path = args.v7_root / "feasibility_gate.json"
    v7_protocol_path = args.v7_root / "protocol_lock.json"
    v7_summary_path = args.v7_root / "model/run_summary.json"
    v7_oof_path = args.v7_root / "model/oof_predictions.jsonl"
    required = [
        raw,
        raw_summary_path,
        train,
        train_manifest_path,
        v5_cache,
        v5_manifest_path,
        v7_gate_path,
        v7_protocol_path,
        v7_summary_path,
        v7_oof_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    missing += [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing)

    raw_summary = load(raw_summary_path)
    train_manifest = load(train_manifest_path)
    v5_manifest = load(v5_manifest_path)
    v7_gate = load(v7_gate_path)
    v7_protocol = load(v7_protocol_path)
    v7_summary = load(v7_summary_path)
    train_rows = read_jsonl(train)
    v7_oof = read_jsonl(v7_oof_path)
    train_ids = [str(row["event_id"]) for row in train_rows]
    train_tasks = {
        str(row["event_id"]): str(row["task_id_hash"]) for row in train_rows
    }
    v7_ids = [str(row["event_id"]) for row in v7_oof]
    recomputed_v7_auc = roc_auc(
        [int(row["label"]) for row in v7_oof],
        [float(row["active_raw_score"]) for row in v7_oof],
    )
    checks = {
        "raw_events_bound": raw_summary.get("event_file_sha256") == file_sha256(raw),
        "train_events_bound": train_manifest.get("source_event_sha256")
        == file_sha256(raw)
        and train_manifest.get("train_sha256") == file_sha256(train),
        "v5_static_features_bound": v5_manifest.get("feature_file_sha256")
        == file_sha256(v5_cache)
        and v5_manifest.get("private_branch_outcomes_cached") is False,
        "v7_one_step_failure_preserved": v7_gate.get("passed") is False,
        "v7_gate_identity": v7_gate.get("stage")
        == "toolsandbox_v7_active_shadow_feasibility_gate_seed42"
        and v7_gate.get("risk_certified") is False
        and bool(v7_gate.get("integrity_checks"))
        and all(v7_gate["integrity_checks"].values()),
        "v7_protocol_identity": v7_protocol.get("status")
        == "frozen_before_toolsandbox_v7_active_shadow"
        and v7_protocol.get("config") == CONFIG
        and v7_protocol.get("gate") == V7_GATE
        and v7_protocol.get("raw_events_sha256") == file_sha256(raw)
        and v7_protocol.get("train_file_sha256") == file_sha256(train)
        and v7_protocol.get("v5_feature_cache_sha256") == file_sha256(v5_cache),
        "v7_summary_identity": v7_summary.get("stage")
        == "toolsandbox_v7_active_shadow_nested_cross_task_oof"
        and v7_summary.get("status") == "completed"
        and v7_summary.get("evaluation_protocol", "").startswith(
            "nested task cross-fitting"
        )
        and v7_summary.get("protocol_lock_sha256")
        == file_sha256(v7_protocol_path)
        and v7_summary.get("oof_predictions_sha256") == file_sha256(v7_oof_path),
        "v7_paired_event_identity": len(train_ids) == len(set(train_ids)) == 126
        and len(v7_ids) == len(set(v7_ids)) == 126
        and set(v7_ids) == set(train_ids)
        and all(
            train_tasks[str(row["event_id"])] == str(row["task_id_hash"])
            for row in v7_oof
        ),
        "v7_auc_recomputed": abs(
            recomputed_v7_auc - float(v7_summary["active_cross_task_roc_auc"])
        )
        <= 1e-12
        and abs(
            recomputed_v7_auc - float(v7_gate["active_cross_task_roc_auc"])
        )
        <= 1e-12,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    protocol = {
        "status": PROTOCOL_STATUS,
        "stage": "toolsandbox_v9_two_step_offline_seed42",
        "checks": checks,
        "config": CONFIG,
        "gate": GATE,
        "raw_events_sha256": file_sha256(raw),
        "train_file_sha256": file_sha256(train),
        "v5_feature_cache_sha256": file_sha256(v5_cache),
        "v7_gate_sha256": file_sha256(v7_gate_path),
        "v7_protocol_sha256": file_sha256(v7_protocol_path),
        "v7_run_summary_sha256": file_sha256(v7_summary_path),
        "v7_oof_predictions_sha256": file_sha256(v7_oof_path),
        "v7_one_step_active_auc": recomputed_v7_auc,
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "probe_horizon": 2,
        "allowed_inputs": [
            "candidate A and B",
            "first two A receipts and executed actions",
            "first two B receipts and executed actions",
            "frozen public-only V5 representation for acquisition",
        ],
        "prohibited_inputs": [
            "official score or score trace",
            "third or later branch receipts",
            "ending context digest",
            "decision label at deployment",
        ],
        "scope": "cross-task offline two-step ActiveShadow feasibility on frozen V4.4 branches; no recollection",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
