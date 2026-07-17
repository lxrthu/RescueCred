#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_task_eval import event_set_hash
from scripts.freeze_route_a_v31_both_valid_protocol import GATE_THRESHOLDS
from scripts.freeze_route_a_v31_confirm_protocol import AGGREGATE_THRESHOLDS


SOURCE_PATHS = (
    "scripts/evaluate_route_a_bounded.py",
    "scripts/freeze_route_a_v31_confirm_bounded_protocol.py",
    "scripts/analyze_route_a_v31_confirm.py",
    "scripts/appworld_azure_continuation_worker.py",
    "scripts/select_route_a_frozen_events.py",
    "scripts/route_a_adapter_scorer_worker.py",
    "rescuecredit/route_a_bounded.py",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(path: Path) -> list[str]:
    return [str(row["event_id"]) for row in read_jsonl(path)]


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
    parser.add_argument("--training-lock", type=Path, required=True)
    parser.add_argument("--preference-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = read_jsonl(args.event_file)
    manifest = _load(args.event_manifest)
    mask_selection, v31_selection = _load(args.mask_selection), _load(args.v31_selection)
    mask_run, v31_run = _load(args.mask_run), _load(args.v31_run)
    training_lock, preference_gate = _load(args.training_lock), _load(args.preference_gate)
    seed = int(training_lock["seed"])
    event_ids = [str(row["event_id"]) for row in events]
    counts = Counter(str(row["task_id"]) for row in events)
    max_share = max(counts.values(), default=0) / max(1, len(events))
    event_hash = event_set_hash(events)
    checks = {
        "training_lock_preoutcome": training_lock.get("status")
        == "frozen_before_v31_outcomes",
        "training_seed_bound": mask_run.get("seed") == v31_run.get("seed") == seed,
        "manifest_frozen": manifest.get("status") == "frozen_before_bounded_outcomes",
        "dev_only": manifest.get("split") == "dev"
        and manifest.get("training_access") is False
        and manifest.get("test_split_access") is False,
        "event_identity": manifest.get("event_set_hash") == event_hash
        and manifest.get("event_file_sha256") == file_sha256(args.event_file),
        "event_diversity": len(events) >= GATE_THRESHOLDS["min_events"]
        and len(counts) >= GATE_THRESHOLDS["min_tasks_with_events"]
        and max_share <= GATE_THRESHOLDS["max_task_event_share"],
        "event_contract": manifest.get("both_actions_schema_complete") is True
        and manifest.get("variant_kinds")
        == {"visible_candidate_value_pair": len(events)},
        "unique_events": len(set(event_ids)) == len(event_ids),
        "selection_coverage": set(_ids(args.mask_results)) == set(event_ids)
        == set(_ids(args.v31_results))
        and len(_ids(args.mask_results)) == len(event_ids)
        and len(_ids(args.v31_results)) == len(event_ids),
        "selection_identity": mask_selection.get("results_sha256")
        == file_sha256(args.mask_results)
        and v31_selection.get("results_sha256") == file_sha256(args.v31_results)
        and mask_selection.get("event_set_hash")
        == v31_selection.get("event_set_hash") == event_hash,
        "selection_methods": mask_selection.get("method") == "mask"
        and v31_selection.get("method") == "v31"
        and mask_selection.get("scoring_failures") == 0
        and v31_selection.get("scoring_failures") == 0,
        "adapter_identity": mask_selection.get("adapter_sha256")
        == mask_run.get("adapter_sha256")
        and v31_selection.get("adapter_sha256") == v31_run.get("adapter_sha256"),
        "preference_gate_bound": preference_gate.get("seed") == seed
        and preference_gate.get("passed") is True
        and preference_gate.get("stage") == f"route_a_v31_confirm_preference_seed{seed}",
    }
    if not all(checks.values()):
        raise RuntimeError(f"confirmatory bounded preflight failed: {checks}")
    lock = {
        "status": "frozen_before_both_valid_confirmatory_outcomes",
        "stage": f"route_a_v31_both_valid_confirm_seed{seed}",
        "seed": seed,
        "horizons": [4, 8],
        "events": len(events),
        "tasks_with_events": len(counts),
        "event_set_hash": event_hash,
        "event_file_sha256": file_sha256(args.event_file),
        "event_manifest_sha256": file_sha256(args.event_manifest),
        "mask_results_sha256": file_sha256(args.mask_results),
        "v31_results_sha256": file_sha256(args.v31_results),
        "mask_run_sha256": file_sha256(args.mask_run),
        "v31_run_sha256": file_sha256(args.v31_run),
        "training_lock_sha256": file_sha256(args.training_lock),
        "preference_gate_sha256": file_sha256(args.preference_gate),
        "method_a": "mask",
        "method_b": "v31",
        "gate_thresholds": GATE_THRESHOLDS,
        "aggregate_thresholds": AGGREGATE_THRESHOLDS,
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "public_openapi_schema_corpus_sha256": directory_sha256(
            Path("data/api_docs/openapi")
        ),
        "checks": checks,
        "reference_boundary": {
            "reference_suffix_exposed_to_continuation": False,
            "test_split_access": False,
        },
        "scope": "same frozen both-valid AppWorld dev fixture; confirmatory training and continuation seed stability only",
    }
    write_json(args.output, lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
