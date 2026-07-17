#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_bounded import summarize_bounded_results


TOLERANCE = 1e-12


def equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= TOLERANCE
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            equivalent(a, b) for a, b in zip(left, right)
        )
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--v31-run", type=Path, required=True)
    parser.add_argument("--preference-gate", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    raw = load(root / "bounded_summary.json")
    original_gate = load(root / "gate.json")
    protocol_path = root / "protocol_lock.json"
    protocol = load(protocol_path)
    rows = read_jsonl(root / "bounded_results.jsonl")
    events = read_jsonl(root / "events/both_valid_dev_events.public.jsonl")
    recomputed = summarize_bounded_results(
        rows,
        horizons=protocol["horizons"],
        event_set_hash=raw["event_set_hash"],
    )
    compared_keys = (
        "events",
        "event_set_hash",
        "horizons",
        "primary_horizon",
        "selection_disagreements",
        "horizon_prefix_mismatches",
        "horizon_prefix_unverifiable",
        "primary",
    )
    failed_identity = sorted(
        key for key, value in original_gate["identity_checks"].items() if not value
    )
    artifact_paths = {
        "event_file_sha256": root / "events/both_valid_dev_events.public.jsonl",
        "event_manifest_sha256": root / "events/manifest.json",
        "mask_results_sha256": root / "mask/task_results.jsonl",
        "v31_results_sha256": root / "v31/task_results.jsonl",
        "mask_selection_sha256": root / "mask/selection_summary.json",
        "v31_selection_sha256": root / "v31/selection_summary.json",
        "mask_run_sha256": args.mask_run,
        "v31_run_sha256": args.v31_run,
        "v31_preference_gate_sha256": args.preference_gate,
    }
    bounded_ids = [str(row["event_id"]) for row in rows]
    event_ids = [str(row["event_id"]) for row in events]
    checks = {
        "original_gate_preserved_as_failure": original_gate.get("passed") is False,
        "only_failed_check_is_roundoff_recompute": failed_identity
        == ["raw_rows_independently_recomputed"],
        "all_original_outcome_checks_pass": all(
            original_gate.get("outcome_checks", {}).values()
        ),
        "external_and_embedded_lock_unchanged": raw.get("protocol_lock") == protocol
        and raw.get("protocol_lock_sha256") == file_sha256(protocol_path),
        "preoutcome_sources_still_match": bool(protocol.get("source_sha256"))
        and all(
            Path(path).is_file() and file_sha256(Path(path)) == expected
            for path, expected in protocol["source_sha256"].items()
        ),
        "all_frozen_artifacts_still_match": all(
            path.is_file() and file_sha256(path) == protocol.get(field)
            for field, path in artifact_paths.items()
        ),
        "bounded_results_unchanged": file_sha256(root / "bounded_results.jsonl")
        == raw.get("bounded_results_sha256")
        and len(rows) == raw.get("bounded_results_rows"),
        "bounded_rows_unique_and_exact": len(rows) == len(events)
        and len(set(bounded_ids)) == len(bounded_ids)
        and set(bounded_ids) == set(event_ids),
        "frozen_horizons_unchanged": protocol.get("horizons") == [4, 8]
        and raw.get("requested_horizons") == [4, 8]
        and raw.get("primary_horizon") == 8,
        "recomputed_metrics_equal_with_1e12_tolerance": all(
            equivalent(raw.get(key), recomputed.get(key)) for key in compared_keys
        ),
        "metrics_and_thresholds_unchanged": original_gate.get("primary_horizon")
        == raw.get("primary_horizon")
        and original_gate.get("thresholds") == protocol.get("gate_thresholds")
        and equivalent(original_gate.get("score_improvement"), raw["primary"]["score_improvement"])
        and equivalent(
            original_gate.get("causal_accuracy_improvement"),
            raw["primary"]["causal_accuracy_improvement"],
        ),
    }
    passed = all(checks.values())
    erratum = copy.deepcopy(original_gate)
    erratum.update(
        {
            "passed": passed,
            "status": "audited_numeric_roundoff_erratum",
            "stage": "route_a_v31_both_valid_appworld_dev_gate_seed42_erratum",
            "erratum_checks": checks,
            "original_gate_sha256": file_sha256(root / "gate.json"),
            "protocol_lock_sha256": file_sha256(protocol_path),
            "roundoff_tolerance": TOLERANCE,
            "metrics_changed": False,
            "thresholds_changed": False,
            "experiment_rerun_required": False,
            "next_step": (
                "freeze confirmatory seeds; retain original gate and this erratum"
                if passed
                else "do not expand seeds; erratum audit failed"
            ),
        }
    )
    output = root / "gate_roundoff_erratum.json"
    write_json(output, erratum)
    print(json.dumps(erratum, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
