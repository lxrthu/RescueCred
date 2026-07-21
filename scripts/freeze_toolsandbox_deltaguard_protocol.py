#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.deltaguard_observers import REGISTRY_VERSION
from rescuecredit.deltaguard_protocol import (
    PROTOCOL_STATUS,
    PUBLIC_HMAC_KEY,
    config_for_role,
    freeze_source_stream,
    load_public_sources,
    source_stream_digest,
)
from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json


SOURCE_PATHS = (
    "rescuecredit/deltaguard_observers.py",
    "rescuecredit/deltaguard_goal_contract.py",
    "rescuecredit/deltaguard_probe.py",
    "rescuecredit/deltaguard_certificate.py",
    "rescuecredit/deltaguard_contract.py",
    "rescuecredit/deltaguard_toolsandbox.py",
    "rescuecredit/deltaguard_evaluation.py",
    "rescuecredit/deltaguard_protocol.py",
    "rescuecredit/deltaguard_baseline.py",
    "scripts/freeze_toolsandbox_deltaguard_protocol.py",
    "scripts/export_toolsandbox_deltaguard_public_bank.py",
    "scripts/collect_toolsandbox_deltaguard.py",
    "scripts/evaluate_toolsandbox_deltaguard.py",
    "scripts/check_toolsandbox_deltaguard_gate.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("sanity", "feasibility", "full"), required=True)
    parser.add_argument("--public-events", type=Path, nargs="+", required=True)
    parser.add_argument("--public-bank-manifest", type=Path, required=True)
    parser.add_argument("--v7-checkpoint", type=Path, required=True)
    parser.add_argument("--v7-train-file", type=Path)
    parser.add_argument("--v7-run-summary", type=Path)
    parser.add_argument("--v7-protocol-lock", type=Path)
    parser.add_argument("--v7-oof", type=Path)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--source-events-per-family", type=int)
    parser.add_argument("--attempt-cap-per-family", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.role == "full" and any(
        value is None
        for value in (args.v7_train_file, args.v7_run_summary, args.v7_protocol_lock)
    ):
        raise ValueError(
            "full role requires V7 train file, run summary, and protocol lock"
        )
    required = [
        *args.public_events,
        args.public_bank_manifest,
        args.v7_checkpoint,
        *(Path(path) for path in SOURCE_PATHS),
    ]
    if args.v7_train_file is not None:
        required.append(args.v7_train_file)
    if args.v7_run_summary is not None:
        required.append(args.v7_run_summary)
    if args.v7_protocol_lock is not None:
        required.append(args.v7_protocol_lock)
    if args.v7_oof is not None:
        required.append(args.v7_oof)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    config = config_for_role(
        args.role,
        families=args.families,
        source_events_per_family=args.source_events_per_family,
        attempt_cap_per_family=args.attempt_cap_per_family,
    )
    public_manifest = json.loads(args.public_bank_manifest.read_text(encoding="utf-8"))
    public_hashes = [file_sha256(path) for path in args.public_events]
    if public_manifest.get("public_bank_sha256") not in public_hashes:
        raise ValueError("public bank manifest does not bind a supplied public bank")
    if public_manifest.get("protected_fields_exported") != []:
        raise ValueError("public bank manifest reports protected exports")
    rows = load_public_sources(args.public_events)
    source_rows, audit = freeze_source_stream(
        rows,
        families=config["families"],
        source_events_per_family=int(config["source_events_per_family"]),
        attempt_cap_per_family=int(config["attempt_cap_per_family"]),
        acquisition_rate=float(config["acquisition_rate"]),
        hmac_key=PUBLIC_HMAC_KEY,
    )
    if not audit["complete"]:
        raise RuntimeError({"source_stream_incomplete": audit})
    source_event_ids = {str(row["event_id"]) for row in source_rows}
    source_tasks = {str(row["task_id_hash"]) for row in source_rows}
    baseline_overlap = {"event_ids": 0, "task_ids": 0, "full_disjoint": None}
    baseline_lineage = {
        "checkpoint_bound": False,
        "protocol_bound": False,
        "train_file_bound": False,
        "valid": False,
    }
    if args.v7_train_file is not None:
        train_rows = read_jsonl(args.v7_train_file)
        train_event_ids = {str(row.get("event_id", "")) for row in train_rows}
        train_tasks = {str(row.get("task_id_hash", "")) for row in train_rows}
        baseline_overlap = {
            "event_ids": len(source_event_ids & train_event_ids),
            "task_ids": len(source_tasks & train_tasks),
            "full_disjoint": not bool(source_event_ids & train_event_ids or source_tasks & train_tasks),
        }
        if args.role == "full" and not baseline_overlap["full_disjoint"]:
            raise RuntimeError({"v7_baseline_source_overlap": baseline_overlap})
    if args.v7_run_summary is not None and args.v7_protocol_lock is not None:
        run_summary = json.loads(args.v7_run_summary.read_text(encoding="utf-8"))
        v7_protocol = json.loads(args.v7_protocol_lock.read_text(encoding="utf-8"))
        baseline_lineage = {
            "checkpoint_bound": run_summary.get("checkpoint_sha256")
            == file_sha256(args.v7_checkpoint),
            "protocol_bound": run_summary.get("protocol_lock_sha256")
            == file_sha256(args.v7_protocol_lock),
            "train_file_bound": args.v7_train_file is not None
            and v7_protocol.get("train_file_sha256") == file_sha256(args.v7_train_file),
            "oof_bound": args.v7_oof is None
            or run_summary.get("oof_predictions_sha256") == file_sha256(args.v7_oof),
            "valid": False,
        }
        baseline_lineage["valid"] = all(
            baseline_lineage[key]
            for key in ("checkpoint_bound", "protocol_bound", "train_file_bound", "oof_bound")
        )
        if args.role == "full" and not baseline_lineage["valid"]:
            raise RuntimeError({"invalid_v7_lineage": baseline_lineage})
    protocol = {
        "status": PROTOCOL_STATUS,
        "stage": f"toolsandbox_deltaguard_{args.role}_protocol",
        "role": args.role,
        "config": config,
        "registry_version": REGISTRY_VERSION,
        "hmac_key_public_commitment": PUBLIC_HMAC_KEY,
        "hmac_key_frozen_before_this_source_bank": True,
        "public_sources": [
            {"path": str(path), "sha256": file_sha256(path)} for path in args.public_events
        ],
        "public_bank_manifest": str(args.public_bank_manifest),
        "public_bank_manifest_sha256": file_sha256(args.public_bank_manifest),
        "v7_checkpoint": str(args.v7_checkpoint),
        "v7_checkpoint_sha256": file_sha256(args.v7_checkpoint),
        "v7_train_file": str(args.v7_train_file) if args.v7_train_file else None,
        "v7_train_file_sha256": file_sha256(args.v7_train_file) if args.v7_train_file else None,
        "v7_baseline_overlap_audit": baseline_overlap,
        "v7_run_summary": str(args.v7_run_summary) if args.v7_run_summary else None,
        "v7_run_summary_sha256": file_sha256(args.v7_run_summary) if args.v7_run_summary else None,
        "v7_protocol_lock": str(args.v7_protocol_lock) if args.v7_protocol_lock else None,
        "v7_protocol_lock_sha256": file_sha256(args.v7_protocol_lock) if args.v7_protocol_lock else None,
        "v7_baseline_lineage": baseline_lineage,
        "v7_oof": str(args.v7_oof) if args.v7_oof else None,
        "v7_oof_sha256": file_sha256(args.v7_oof) if args.v7_oof else None,
        "v7_hash_dimension": int(
            json.loads(args.v7_protocol_lock.read_text(encoding="utf-8"))["config"]["hash_dimension"]
            if args.v7_protocol_lock
            else 256
        ),
        "source_events": source_rows,
        "source_stream_sha256": source_stream_digest(source_rows),
        "source_audit": audit,
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "labels_available_to_freezer": False,
        "conditioning_scope": public_manifest.get("conditioning_scope"),
        "protected_inputs": [
            "decision",
            "official score",
            "reference action",
            "full branch return",
            "hidden database/context",
        ],
        "estimands": {
            "conditional": "ROC-AUC on selected public paired probes",
            "whole_stream": "Reverse recall, Rescue drop, and probe rate over all frozen source events",
        },
        "formal_risk_claim_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
