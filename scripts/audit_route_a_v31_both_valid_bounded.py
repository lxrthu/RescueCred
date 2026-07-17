#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_bounded import summarize_bounded_results
from rescuecredit.route_a_task_eval import event_set_hash
from scripts.freeze_route_a_v31_both_valid_protocol import GATE_THRESHOLDS


TOLERANCE = 1e-12


def _audit_equivalent(left: Any, right: Any) -> bool:
    """Exact structural comparison with tolerance only for numeric roundoff."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= TOLERANCE
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _audit_equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _audit_equivalent(a, b) for a, b in zip(left, right)
        )
    return left == right


def _rename_v2(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key.replace("v2", "v31"): _rename_v2(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_v2(item) for item in value]
    if isinstance(value, str):
        return value.replace("v2", "v31")
    return value


def build_audit(
    *,
    raw_summary: dict[str, Any],
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    mask_selection: dict[str, Any],
    v31_selection: dict[str, Any],
    preference_gate: dict[str, Any],
    source_identity_matches: bool,
    protocol_lock_sha256: str,
    artifact_identity_matches: bool,
    recomputed: dict[str, Any],
    horizon_binding_matches: bool,
    bounded_row_identity_matches: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = raw_summary["primary"]
    identity_checks = {
        "protocol_validated_by_evaluator": raw_summary.get(
            "protocol_lock_validated"
        )
        is True
        and raw_summary.get("development_protocol") is True,
        "external_and_embedded_protocol_match": raw_summary.get(
            "protocol_lock_sha256"
        )
        == protocol_lock_sha256
        and raw_summary.get("protocol_lock") == protocol,
        "protocol_status_and_methods": protocol.get("status")
        == "frozen_before_both_valid_dev_outcomes"
        and protocol.get("method_a") == "mask"
        and protocol.get("method_b") == "v31",
        "event_identity_bound": raw_summary.get("event_set_hash")
        == protocol.get("event_set_hash")
        == manifest.get("event_set_hash")
        and raw_summary.get("event_file_sha256")
        == protocol.get("event_file_sha256")
        == manifest.get("event_file_sha256"),
        "both_valid_event_contract": manifest.get(
            "both_actions_schema_complete"
        )
        is True
        and manifest.get("variant_kinds")
        == {"visible_candidate_value_pair": manifest.get("events")},
        "event_coverage_and_diversity": int(manifest.get("events", 0))
        >= GATE_THRESHOLDS["min_events"]
        and int(manifest.get("tasks_with_events", 0))
        >= GATE_THRESHOLDS["min_tasks_with_events"]
        and float(manifest.get("max_task_event_share", 1.0))
        <= GATE_THRESHOLDS["max_task_event_share"],
        "selection_methods_and_results_bound": mask_selection.get("method")
        == "mask"
        and v31_selection.get("method") == "v31"
        and raw_summary.get("mask_results_sha256")
        == protocol.get("mask_results_sha256")
        == mask_selection.get("results_sha256")
        and raw_summary.get("v2_results_sha256")
        == protocol.get("v31_results_sha256")
        == v31_selection.get("results_sha256"),
        "selection_scoring_succeeded": mask_selection.get("scoring_failures")
        == 0
        and v31_selection.get("scoring_failures") == 0,
        "preference_gate_bound": preference_gate.get("passed") is True
        and preference_gate.get("stage")
        == "route_a_seed42_v31_validity_first_gate",
        "source_identity_frozen": source_identity_matches,
        "all_frozen_artifacts_bound": artifact_identity_matches,
        "frozen_horizons_and_primary_bound": horizon_binding_matches,
        "bounded_rows_unique_and_exact": bounded_row_identity_matches,
        "raw_rows_independently_recomputed": all(
            _audit_equivalent(raw_summary.get(key), recomputed.get(key))
            for key in (
                "events",
                "event_set_hash",
                "horizons",
                "primary_horizon",
                "selection_disagreements",
                "horizon_prefix_mismatches",
                "horizon_prefix_unverifiable",
                "primary",
            )
        ),
        "continuation_reference_boundary": raw_summary.get(
            "continuation_input_excludes_evaluator_and_reference"
        )
        is True
        and raw_summary.get("reference_suffix_used") is False
        and raw_summary.get("test_split_access") is False,
        "cache_and_prefix_integrity": int(raw_summary.get("cache_conflicts", -1))
        == 0
        and int(raw_summary.get("horizon_prefix_mismatches", -1)) == 0
        and int(raw_summary.get("horizon_prefix_unverifiable", -1)) == 0,
        "full_run_not_sanity": raw_summary.get("sanity_limit") is None,
        "frozen_thresholds": protocol.get("gate_thresholds") == GATE_THRESHOLDS,
    }
    outcome_checks = {
        "enough_valid_paired_events": int(primary["valid_paired_events"])
        >= GATE_THRESHOLDS["min_valid_paired_events"],
        "enough_nonzero_causal_events": int(primary["nonzero_causal_events"])
        >= GATE_THRESHOLDS["min_nonzero_causal_events"],
        "methods_make_different_selections": int(
            raw_summary["selection_disagreements"]
        )
        >= GATE_THRESHOLDS["min_selection_disagreements"],
        "v31_improves_bounded_official_score": float(primary["score_improvement"])
        > TOLERANCE,
        "v31_has_more_wins_than_losses": int(primary["v2_better_events"])
        > int(primary["v2_worse_events"]),
        "v31_improves_causal_selection_accuracy": float(
            primary["causal_accuracy_improvement"]
        )
        > TOLERANCE,
    }
    passed = all(identity_checks.values()) and all(outcome_checks.values())
    summary = _rename_v2(raw_summary)
    summary.update(
        {
            "stage": "route_a_v31_both_valid_appworld_dev_seed42",
            "method_a": "mask",
            "method_b": "v31",
            "method_b_was_legacy_v2_cli_slot": True,
            "identity_checks": identity_checks,
            "scope": protocol["scope"],
        }
    )
    gate = {
        "passed": passed,
        "stage": "route_a_v31_both_valid_appworld_dev_gate_seed42",
        "identity_checks": identity_checks,
        "outcome_checks": outcome_checks,
        "thresholds": GATE_THRESHOLDS,
        "events": manifest["events"],
        "tasks_with_events": manifest["tasks_with_events"],
        "primary_horizon": raw_summary["primary_horizon"],
        "selection_disagreements": raw_summary["selection_disagreements"],
        "valid_paired_events": primary["valid_paired_events"],
        "nonzero_causal_events": primary["nonzero_causal_events"],
        "mask_mean_official_score": primary["mask_mean_official_score"],
        "v31_mean_official_score": primary["v2_mean_official_score"],
        "score_improvement": primary["score_improvement"],
        "mask_causal_selection_accuracy": primary[
            "mask_causal_selection_accuracy"
        ],
        "v31_causal_selection_accuracy": primary[
            "v2_causal_selection_accuracy"
        ],
        "causal_accuracy_improvement": primary[
            "causal_accuracy_improvement"
        ],
        "v31_better_events": primary["v2_better_events"],
        "v31_worse_events": primary["v2_worse_events"],
        "ties": primary["ties"],
        "scope": protocol["scope"],
        "next_step": (
            "freeze confirmatory training seeds and an untouched evaluation partition"
            if passed
            else "do not expand seeds; inspect both-valid dev failures"
        ),
    }
    return summary, gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-summary", type=Path, required=True)
    parser.add_argument("--bounded-results", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("--event-manifest", type=Path, required=True)
    parser.add_argument("--mask-results", type=Path, required=True)
    parser.add_argument("--v31-results", type=Path, required=True)
    parser.add_argument("--mask-selection", type=Path, required=True)
    parser.add_argument("--v31-selection", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--v31-run", type=Path, required=True)
    parser.add_argument("--preference-gate", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    protocol = load(args.protocol_lock)
    raw_summary = load(args.raw_summary)
    bounded_rows = read_jsonl(args.bounded_results)
    frozen_events = read_jsonl(args.event_file)
    recomputed = summarize_bounded_results(
        bounded_rows,
        horizons=list(protocol.get("horizons", [])),
        event_set_hash=raw_summary.get("event_set_hash", ""),
    )
    horizon_binding_matches = (
        raw_summary.get("requested_horizons") == protocol.get("horizons") == [4, 8]
        and raw_summary.get("primary_horizon") == max(protocol.get("horizons", [0]))
    )
    bounded_ids = [str(row["event_id"]) for row in bounded_rows]
    frozen_ids = [str(row["event_id"]) for row in frozen_events]
    bounded_row_identity_matches = (
        len(bounded_rows) == len(frozen_events)
        and len(set(bounded_ids)) == len(bounded_ids)
        and set(bounded_ids) == set(frozen_ids)
    )
    source_identity_matches = bool(protocol.get("source_sha256")) and all(
        Path(path).is_file() and file_sha256(Path(path)) == expected
        for path, expected in protocol.get("source_sha256", {}).items()
    ) and directory_sha256(Path("data/api_docs/openapi")) == protocol.get(
        "public_openapi_schema_corpus_sha256"
    )
    artifact_identity_matches = all(
        (
            actual.is_file()
            and file_sha256(actual) == protocol.get(field)
        )
        for actual, field in (
            (args.event_file, "event_file_sha256"),
            (args.event_manifest, "event_manifest_sha256"),
            (args.mask_results, "mask_results_sha256"),
            (args.v31_results, "v31_results_sha256"),
            (args.mask_selection, "mask_selection_sha256"),
            (args.v31_selection, "v31_selection_sha256"),
            (args.mask_run, "mask_run_sha256"),
            (args.v31_run, "v31_run_sha256"),
            (args.preference_gate, "v31_preference_gate_sha256"),
        )
    ) and raw_summary.get("bounded_results_sha256") == file_sha256(
        args.bounded_results
    ) and raw_summary.get("bounded_results_rows") == len(bounded_rows)
    artifact_identity_matches = artifact_identity_matches and raw_summary.get(
        "bounded_results_event_set_hash"
    ) == event_set_hash(bounded_rows)
    summary, gate = build_audit(
        raw_summary=raw_summary,
        protocol=protocol,
        manifest=load(args.event_manifest),
        mask_selection=load(args.mask_selection),
        v31_selection=load(args.v31_selection),
        preference_gate=load(args.preference_gate),
        source_identity_matches=source_identity_matches,
        protocol_lock_sha256=file_sha256(args.protocol_lock),
        artifact_identity_matches=artifact_identity_matches,
        recomputed=recomputed,
        horizon_binding_matches=horizon_binding_matches,
        bounded_row_identity_matches=bounded_row_identity_matches,
    )
    write_json(args.summary_output, summary)
    write_json(args.gate_output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
