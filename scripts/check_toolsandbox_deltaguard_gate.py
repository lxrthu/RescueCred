#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.deltaguard_baseline import compute_v7_baseline_scores
from rescuecredit.deltaguard_evaluation import evaluate_deltaguard, label_from_decision
from rescuecredit.deltaguard_protocol import (
    PROTOCOL_STATUS,
    export_public_event,
    load_public_sources,
    verify_protocol_source_identity,
)
from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--label-events", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = _load(args.protocol_lock)
    verify_protocol_source_identity(protocol)
    summary_path = args.evaluation_dir / "evaluation_summary.json"
    predictions_path = args.evaluation_dir / "predictions.jsonl"
    source_path = args.collection_dir / "source_ledger.jsonl"
    probe_path = args.collection_dir / "probe_ledger.jsonl"
    manifest_path = args.collection_dir / "collection_manifest.json"
    summary = _load(summary_path)
    manifest = _load(manifest_path)
    source_rows = read_jsonl(source_path)
    probe_rows = read_jsonl(probe_path)
    predictions = read_jsonl(predictions_path)
    label_rows = []
    for path in args.label_events:
        label_rows.extend(read_jsonl(path))
    raw_by_id = {str(row["event_id"]): row for row in label_rows}
    if len(raw_by_id) != len(label_rows):
        raise ValueError("label banks contain duplicate event IDs")
    public_by_id = {
        str(row["event_id"]): row
        for row in load_public_sources(
            [Path(item["path"]) for item in protocol["public_sources"]]
        )
    }
    labels = {
        str(row["event_id"]): label_from_decision(str(raw_by_id[str(row["event_id"])]["decision"]))
        for row in source_rows
    }
    public_manifest = _load(Path(protocol["public_bank_manifest"]))
    baseline_scores, baseline_sources = compute_v7_baseline_scores(
        probe_rows=probe_rows,
        checkpoint_path=Path(protocol["v7_checkpoint"]),
        hash_dimension=int(protocol["v7_hash_dimension"]),
        oof_path=Path(protocol["v7_oof"]) if protocol.get("v7_oof") else None,
    )
    config = protocol["config"]
    recomputed = evaluate_deltaguard(
        source_rows=source_rows,
        probe_rows=probe_rows,
        labels=labels,
        baseline_scores=baseline_scores,
        min_class_per_family=int(config["min_class_per_family"]),
        min_auc=float(config["min_typed_delta_roc_auc"]),
        min_auc_gain=float(config["min_auc_gain_over_v7"]),
        max_probe_rate=float(config["max_probe_rate"]),
        alpha=float(config["risk_alpha"]),
    )
    integrity = {
        "protocol_status": protocol.get("status") == PROTOCOL_STATUS,
        "collection_bound": manifest.get("protocol_lock_sha256") == file_sha256(args.protocol_lock),
        "source_ledger_bound": manifest.get("source_ledger_sha256") == file_sha256(source_path),
        "probe_ledger_bound": manifest.get("probe_ledger_sha256") == file_sha256(probe_path),
        "evaluation_bound": summary.get("protocol_lock_sha256") == file_sha256(args.protocol_lock),
        "predictions_bound": summary.get("predictions_sha256") == file_sha256(predictions_path),
        "metrics_recomputed": all(
            recomputed.get(key) == summary.get(key)
            for key in (
                "status",
                "inconclusive_reasons",
                "conditional_discriminability",
                "whole_stream_public_paired_deltas",
                "whole_stream_contract_abstention",
                "contract_retention",
                "feasibility_passed",
            )
        ),
        "labels_hidden_during_collection": manifest.get("labels_read") is False,
        "public_only_collection": manifest.get("official_evaluator_called") is False
        and manifest.get("hidden_state_exported") is False,
        "formal_risk_claim_absent": summary.get("formal_risk_claim_made") is False,
        "baseline_recomputed_from_frozen_artifacts": all(
            row.get("v7_score") is None
            or abs(float(row["v7_score"]) - baseline_scores[str(row["event_id"])]) < 1e-12
            for row in predictions
        )
        and all(
            row.get("v7_score_source") is None
            or row.get("v7_score_source") == baseline_sources[str(row["event_id"])]
            for row in predictions
        ),
        "label_sources_bound_after_collection": summary.get("label_source_sha256")
        == [file_sha256(path) for path in args.label_events],
        "label_sources_match_public_bank_seal": public_manifest.get("raw_source_sha256")
        == [file_sha256(path) for path in args.label_events],
        "public_projection_matches_sealed_labels": all(
            str(row["event_id"]) in raw_by_id
            and str(row["event_id"]) in public_by_id
            and public_by_id[str(row["event_id"])]
            == export_public_event(raw_by_id[str(row["event_id"])])
            for row in source_rows
        ),
        "exact_replay_labels_only": all(
            raw_by_id[str(row["event_id"])].get("replay_valid") is True
            for row in source_rows
        ),
        "no_collection_errors": manifest.get("successful_probes")
        == manifest.get("probe_events"),
        "normalized_prefix_replay": all(
            row.get("collection_error") is None
            and row.get("replay_audit", {}).get("normalized_visible_history_equal") is True
            for row in probe_rows
        ),
        "observer_plan_and_prefix_valid": all(
            row.get("certificate", {}).get("prefix_unchanged") is True
            for row in probe_rows
        ),
        "full_baseline_lineage_valid": protocol.get("role") != "full"
        or protocol.get("v7_baseline_lineage", {}).get("valid") is True,
        "full_source_disjoint_from_v7": protocol.get("role") != "full"
        or protocol.get("v7_baseline_overlap_audit", {}).get("full_disjoint") is True,
    }
    passed = bool(all(integrity.values()) and recomputed["feasibility_passed"])
    paper_claim_supported = bool(protocol.get("role") == "full" and passed)
    gate = {
        "stage": "toolsandbox_deltaguard_feasibility_gate",
        "passed": passed,
        "paper_claim_supported": paper_claim_supported,
        "role": protocol.get("role"),
        "status": recomputed["status"],
        "integrity_checks": integrity,
        "outcome": recomputed,
        "collection_costs": {
            key: manifest.get(key)
            for key in (
                "observer_calls",
                "observer_calls_per_successful_probe",
                "branch_action_executions",
                "prefix_actions_replayed",
                "unknown_rate",
                "rollback_failures",
                "collection_errors",
            )
        },
        "next_step": (
            "freeze an independent iid episode-level Rescue certification stream"
            if paper_claim_supported
            else "run the fixed 240-event full pilot"
            if passed
            else "stop DeltaGuard only if the full fixed pilot is completed and misses the gate"
        ),
        "claim_boundary": (
            "full public-delta feasibility supported; no finite-sample Rescue-risk claim"
            if paper_claim_supported
            else "sanity/feasibility role only; no paper-facing positive claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
