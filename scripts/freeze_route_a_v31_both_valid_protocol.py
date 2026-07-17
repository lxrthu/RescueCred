#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_task_eval import event_set_hash
from scripts.build_route_a_both_valid_dev_events import _schema_valid


GATE_THRESHOLDS = {
    "min_events": 30,
    "min_tasks_with_events": 15,
    "max_task_event_share": 0.10,
    "max_reference_execution_failure_rate": 0.05,
    "min_valid_paired_events": 30,
    "min_nonzero_causal_events": 5,
    "min_selection_disagreements": 3,
    "require_positive_score_improvement": True,
    "require_more_wins_than_losses": True,
    "require_positive_causal_accuracy_improvement": True,
}


SOURCE_PATHS = (
    "scripts/build_route_a_both_valid_dev_events.py",
    "scripts/select_route_a_frozen_events.py",
    "scripts/evaluate_route_a_bounded.py",
    "scripts/freeze_route_a_v31_both_valid_protocol.py",
    "scripts/audit_route_a_v31_both_valid_bounded.py",
    "scripts/appworld_azure_continuation_worker.py",
    "scripts/appworld_azure_candidate_selector_worker.py",
    "scripts/route_a_adapter_scorer_worker.py",
    "scripts/train_route_a_preference.py",
    "scripts/build_appworld_route_a_bank_v21.py",
    "scripts/audit_appworld_deployable_harness.py",
    "scripts/cloud/run_route_a_v31_both_valid_dev_seed42.sh",
    "environments/appworld/adapter.py",
    "environments/appworld/deployable.py",
    "rescuecredit/route_a_bounded.py",
    "rescuecredit/route_a_preference.py",
    "rescuecredit/appworld_shadow_credit.py",
    "rescuecredit/azure_client.py",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(path: Path) -> list[str]:
    return [str(row["event_id"]) for row in read_jsonl(path)]


def _single_parameter_difference(row: dict[str, Any]) -> bool:
    action_a = row.get("action_a", {})
    action_b = row.get("action_b", {})
    if action_a.get("tool") != action_b.get("tool"):
        return False
    args_a = action_a.get("arguments")
    args_b = action_b.get("arguments")
    if not isinstance(args_a, dict) or not isinstance(args_b, dict):
        return False
    differences = [
        key for key in set(args_a) | set(args_b) if args_a.get(key) != args_b.get(key)
    ]
    return differences == [row.get("parameter")]


def _schema_complete(row: dict[str, Any]) -> bool:
    public_schema = {
        "required_fields": row.get("required_fields", []),
        "parameter_schemas": row.get("parameter_schemas", {}),
        "unsupported_schema_keywords": row.get(
            "unsupported_schema_keywords", []
        ),
    }
    return all(
        row.get(f"{name}_schema_valid") is True
        and _schema_valid(row.get(name, {}), public_schema)
        for name in ("action_a", "action_b")
    )


def _common_pretreatment_context(row: dict[str, Any]) -> bool:
    try:
        context = json.loads(row.get("continuation_context", ""))
    except (TypeError, json.JSONDecodeError):
        return False
    return (
        context.get("original_visible_proposal") == row.get("action_a")
        and "action_b" not in context
        and "reference_suffix" not in context
        and "outcome" not in context
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("--event-manifest", type=Path, required=True)
    parser.add_argument("--mask-results", type=Path, required=True)
    parser.add_argument("--v31-results", type=Path, required=True)
    parser.add_argument("--mask-selection", type=Path, required=True)
    parser.add_argument("--v31-selection", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--v31-run", type=Path, required=True)
    parser.add_argument("--v31-preference-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = read_jsonl(args.event_file)
    manifest = _load(args.event_manifest)
    mask_selection = _load(args.mask_selection)
    v31_selection = _load(args.v31_selection)
    mask_run = _load(args.mask_run)
    v31_run = _load(args.v31_run)
    preference_gate = _load(args.v31_preference_gate)
    event_ids = [str(row["event_id"]) for row in events]
    task_counts = Counter(str(row["task_id"]) for row in events)
    max_task_share = max(task_counts.values(), default=0) / max(1, len(events))
    event_hash = event_set_hash(events)
    event_sha = file_sha256(args.event_file)
    checks = {
        "manifest_frozen_before_outcomes": manifest.get("status")
        == "frozen_before_bounded_outcomes",
        "dev_only_no_test_or_training": manifest.get("split") == "dev"
        and manifest.get("training_access") is False
        and manifest.get("test_split_access") is False,
        "enough_events": len(events) >= GATE_THRESHOLDS["min_events"],
        "enough_tasks": len(task_counts)
        >= GATE_THRESHOLDS["min_tasks_with_events"],
        "not_task_dominated": max_task_share
        <= GATE_THRESHOLDS["max_task_event_share"],
        "reference_replay_healthy": float(
            manifest.get("reference_execution_failure_rate", 1.0)
        )
        <= GATE_THRESHOLDS["max_reference_execution_failure_rate"],
        "manifest_identity": manifest.get("events") == len(events)
        and manifest.get("event_set_hash") == event_hash
        and manifest.get("event_file_sha256") == event_sha,
        "unique_event_ids": len(set(event_ids)) == len(event_ids),
        "all_events_both_valid_by_public_contract": all(
            row.get("variant_kind") == "visible_candidate_value_pair"
            and _schema_complete(row)
            and _single_parameter_difference(row)
            and _common_pretreatment_context(row)
            for row in events
        ),
        "no_outcome_labels_exported": manifest.get(
            "protected_outcome_labels_exported"
        )
        is False,
        "selection_coverage_exact": set(_ids(args.mask_results)) == set(event_ids)
        == set(_ids(args.v31_results))
        and len(_ids(args.mask_results)) == len(event_ids)
        and len(_ids(args.v31_results)) == len(event_ids),
        "selection_methods_bound": mask_selection.get("method") == "mask"
        and v31_selection.get("method") == "v31"
        and mask_selection.get("results_sha256")
        == file_sha256(args.mask_results)
        and v31_selection.get("results_sha256")
        == file_sha256(args.v31_results),
        "selection_event_identity": mask_selection.get("event_set_hash")
        == v31_selection.get("event_set_hash")
        == event_hash
        and mask_selection.get("event_file_sha256")
        == v31_selection.get("event_file_sha256")
        == event_sha,
        "selection_scoring_succeeded": mask_selection.get("scoring_failures") == 0
        and v31_selection.get("scoring_failures") == 0,
        "adapters_bound": mask_selection.get("adapter_sha256")
        == mask_run.get("adapter_sha256")
        and v31_selection.get("adapter_sha256") == v31_run.get("adapter_sha256"),
        "base_model_and_scorer_bound": mask_selection.get("base_model_sha256")
        == v31_selection.get("base_model_sha256")
        == mask_run.get("base_model_sha256")
        == v31_run.get("base_model_sha256")
        and mask_selection.get("scorer_script_sha256")
        == v31_selection.get("scorer_script_sha256")
        and mask_selection.get("runtime_identity")
        == v31_selection.get("runtime_identity"),
        "v31_preference_gate_passed": preference_gate.get("passed") is True
        and preference_gate.get("stage")
        == "route_a_seed42_v31_validity_first_gate",
        "v31_training_bound": v31_run.get("method") == "v31"
        and v31_run.get("validity_first") is True
        and mask_run.get("method") == "mask",
    }
    if not all(checks.values()):
        raise SystemExit(
            json.dumps({"passed": False, "checks": checks}, indent=2)
        )

    source_sha256 = {
        path: file_sha256(Path(path))
        for path in SOURCE_PATHS
    }
    lock = {
        "status": "frozen_before_both_valid_dev_outcomes",
        "stage": "route_a_v31_both_valid_appworld_dev_seed42",
        "seed": 42,
        "horizons": [4, 8],
        "events": len(events),
        "tasks_with_events": len(task_counts),
        "max_task_event_share": max_task_share,
        "event_set_hash": event_hash,
        "event_file_sha256": event_sha,
        "event_manifest_sha256": file_sha256(args.event_manifest),
        "mask_results_sha256": file_sha256(args.mask_results),
        "v31_results_sha256": file_sha256(args.v31_results),
        "mask_selection_sha256": file_sha256(args.mask_selection),
        "v31_selection_sha256": file_sha256(args.v31_selection),
        "mask_run_sha256": file_sha256(args.mask_run),
        "v31_run_sha256": file_sha256(args.v31_run),
        "v31_preference_gate_sha256": file_sha256(args.v31_preference_gate),
        "mask_adapter_sha256": mask_run["adapter_sha256"],
        "v31_adapter_sha256": v31_run["adapter_sha256"],
        "base_model_sha256": mask_selection["base_model_sha256"],
        "scorer_script_sha256": mask_selection["scorer_script_sha256"],
        "selection_runtime_identity": mask_selection["runtime_identity"],
        "method_a": "mask",
        "method_b": "v31",
        "gate_thresholds": GATE_THRESHOLDS,
        "source_sha256": source_sha256,
        "public_openapi_schema_corpus_sha256": directory_sha256(
            Path("data/api_docs/openapi")
        ),
        "checks": checks,
        "reference_boundary": {
            "controlled_reference_prefix_and_common_action_fields": True,
            "original_proposal_a_is_common_pretreatment_context": True,
            "reference_action_used_as_preference_label": False,
            "reference_suffix_exposed_to_continuation": False,
            "test_split_access": False,
        },
        "scope": (
            "proposal-conditioned controlled-state AppWorld repair diagnostic in "
            "which both alternatives satisfy public OpenAPI schemas; not neutral "
            "pairwise ranking and not fully autonomous task success"
        ),
    }
    write_json(args.output, lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
