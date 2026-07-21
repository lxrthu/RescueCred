#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rescuecredit.deltaguard_evaluation import label_from_decision
from rescuecredit.deltaguard_protocol import (
    export_public_event,
    load_public_sources,
)
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.goal_directed_query import QUERY_VERSION
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_active_shadow import exact_binomial_upper_bound
from scripts.freeze_toolsandbox_goal_query_pilot import PROTOCOL_STATUS


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(rows: list[dict[str, Any]], route_key: str) -> dict[str, Any]:
    rescue = [row for row in rows if int(row["label"]) == 0]
    reverse = [row for row in rows if int(row["label"]) == 1]
    routes = [row for row in rows if bool(row[route_key])]
    harms = sum(int(row["label"]) == 0 for row in routes)
    hits = sum(int(row["label"]) == 1 for row in routes)
    return {
        "events": len(rows),
        "rescue_events": len(rescue),
        "reverse_events": len(reverse),
        "route_to_a": len(routes),
        "rescue_harms": harms,
        "rescue_drop": harms / len(rescue) if rescue else None,
        "rescue_risk_upper_bound": (
            exact_binomial_upper_bound(harms, len(rescue), alpha=0.05)
            if rescue
            else None
        ),
        "reverse_hits": hits,
        "reverse_recall": hits / len(reverse) if reverse else None,
        "route_precision_reverse": hits / len(routes) if routes else None,
        "witness_coverage": len(routes) / len(rows) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--label-events", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protocol = _load(args.protocol_lock)
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("invalid goal-query protocol")
    collection_manifest_path = args.collection_dir / "collection_manifest.json"
    ledger_path = args.collection_dir / "query_ledger.jsonl"
    collection = _load(collection_manifest_path)
    ledger = read_jsonl(ledger_path)
    if collection.get("protocol_lock_sha256") != file_sha256(args.protocol_lock):
        raise ValueError("goal-query collection/protocol mismatch")
    if collection.get("query_ledger_sha256") != file_sha256(ledger_path):
        raise ValueError("goal-query collection ledger drift")

    public_manifest = _load(Path(protocol["public_bank_manifest"]))
    label_hashes = [file_sha256(path) for path in args.label_events]
    if label_hashes != public_manifest.get("raw_source_sha256"):
        raise ValueError("goal-query label sources do not match public-bank seal")
    raw_rows = []
    for path in args.label_events:
        raw_rows.extend(read_jsonl(path))
    raw_by_id = {str(row["event_id"]): row for row in raw_rows}
    if len(raw_by_id) != len(raw_rows):
        raise ValueError("goal-query label sources contain duplicate events")
    public_by_id = {
        str(row["event_id"]): row
        for row in load_public_sources(
            [Path(item["path"]) for item in protocol["public_sources"]]
        )
    }
    frozen_by_id = {
        str(row["event_id"]): row for row in protocol["source_events"]
    }
    ledger_by_id = {str(row["event_id"]): row for row in ledger}
    if set(frozen_by_id) != set(ledger_by_id):
        raise ValueError("goal-query ledger event set mismatch")

    rows = []
    projection_matches = True
    certificate_valid = True
    for event_id, frozen in frozen_by_id.items():
        raw = raw_by_id.get(event_id)
        public = public_by_id.get(event_id)
        collected = ledger_by_id[event_id]
        if raw is None or public is None:
            raise ValueError(f"goal-query identity missing: {event_id}")
        projection_matches = projection_matches and export_public_event(raw) == public
        if raw.get("replay_valid") is not True:
            raise ValueError("goal-query label is not exact-replay valid")
        certificate = collected.get("certificate")
        certificate_valid = certificate_valid and (
            collected.get("collection_error") is None
            and isinstance(certificate, dict)
            and certificate.get("version") == QUERY_VERSION
            and certificate.get("labels_read") is False
            and bool(certificate.get("route_to_a"))
            == bool(collected.get("route_to_a"))
            and bool(certificate.get("schema_only_route_to_a"))
            == bool(collected.get("schema_only_route_to_a"))
            and bool(certificate.get("query_incremental_route_to_a"))
            == bool(collected.get("query_incremental_route_to_a"))
        )
        rows.append(
            {
                "event_id": event_id,
                "task_id_hash": str(frozen["task_id_hash"]),
                "family": str(frozen["family"]),
                "label": label_from_decision(str(raw["decision"])),
                "route_to_a": bool(collected["route_to_a"]),
                "schema_only_route_to_a": bool(
                    collected["schema_only_route_to_a"]
                ),
                "query_incremental_route_to_a": bool(
                    collected["query_incremental_route_to_a"]
                ),
                "query_executed": collected.get("query_receipt") is not None,
                "witness_reasons": (
                    certificate.get("witness_reasons", [])
                    if isinstance(certificate, dict)
                    else []
                ),
            }
        )

    full = _metrics(rows, "route_to_a")
    schema_only = _metrics(rows, "schema_only_route_to_a")
    query_incremental_hits = sum(
        row["label"] == 1 and row["query_incremental_route_to_a"] for row in rows
    )
    query_incremental_harms = sum(
        row["label"] == 0 and row["query_incremental_route_to_a"] for row in rows
    )
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_rows[row["family"]].append(row)
    by_family = {
        family: _metrics(fold, "route_to_a")
        for family, fold in sorted(family_rows.items())
    }
    query_calls = int(collection["query_calls"])
    integrity = {
        "protocol_bound": collection.get("protocol_lock_sha256")
        == file_sha256(args.protocol_lock),
        "collection_bound": collection.get("query_ledger_sha256")
        == file_sha256(ledger_path),
        "public_projection_matches_sealed_labels": projection_matches,
        "labels_hidden_during_collection": collection.get("labels_read") is False,
        "no_a_or_b_branch_execution": collection.get("a_branch_executions") == 0
        and collection.get("b_branch_executions") == 0,
        "no_hidden_evaluator": collection.get("official_evaluator_called") is False
        and collection.get("hidden_state_exported") is False,
        "no_collection_errors": collection.get("collection_errors") == {},
        "certificates_recomputed_consistent": certificate_valid,
        "event_set_bound": len(rows) == len(protocol["source_events"]),
    }
    gate = protocol["gate"]
    outcomes = {
        "minimum_events": len(rows) >= int(gate["min_events"]),
        "minimum_rescue_events": full["rescue_events"]
        >= int(gate["min_rescue_events"]),
        "minimum_reverse_events": full["reverse_events"]
        >= int(gate["min_reverse_events"]),
        "empirical_rescue_budget": full["rescue_drop"] is not None
        and float(full["rescue_drop"])
        <= float(gate["max_empirical_rescue_drop"]),
        "reverse_recall": full["reverse_recall"] is not None
        and float(full["reverse_recall"]) >= float(gate["min_reverse_recall"]),
        "query_adds_reverse_evidence": query_incremental_hits
        >= int(gate["min_query_incremental_reverse_hits"]),
        "query_cost": query_calls / len(rows)
        <= float(gate["max_queries_per_event"]),
    }
    passed = all(integrity.values()) and all(outcomes.values())
    summary = {
        "status": "passed" if passed else "failed",
        "stage": "toolsandbox_goal_directed_query_pilot0_gate",
        "passed": passed,
        "integrity_checks": integrity,
        "outcome_checks": outcomes,
        "thresholds": gate,
        "observed": {
            "full": full,
            "schema_only": schema_only,
            "query_incremental_reverse_hits": query_incremental_hits,
            "query_incremental_rescue_harms": query_incremental_harms,
            "query_calls": query_calls,
            "queries_per_event": query_calls / len(rows),
            "families": by_family,
        },
        "formal_risk_certified": False,
        "paper_claim_supported": False,
        "claim_boundary": protocol["claim_boundary"],
        "next_step": (
            "freeze a fresh task-disjoint confirmation with deployment-rate accounting"
            if passed
            else "stop Goal-Directed ActiveShadow; the minimal query channel did not clear feasibility"
        ),
        "artifact_hashes": {
            "protocol_lock_sha256": file_sha256(args.protocol_lock),
            "collection_manifest_sha256": file_sha256(collection_manifest_path),
            "query_ledger_sha256": file_sha256(ledger_path),
            "label_source_sha256": label_hashes,
        },
    }
    args.output_dir.mkdir(parents=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, rows)
    summary["artifact_hashes"]["predictions_sha256"] = file_sha256(
        predictions_path
    )
    write_json(args.output_dir / "feasibility_gate.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
