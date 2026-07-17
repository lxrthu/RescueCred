#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import file_sha256
from rescuecredit.logging import write_json


def _rename_v2(value: Any) -> Any:
    if isinstance(value, dict):
        renamed: dict[str, Any] = {}
        for key, item in value.items():
            new_key = key.replace("v2", "v3")
            renamed[new_key] = _rename_v2(item)
        return renamed
    if isinstance(value, list):
        return [_rename_v2(item) for item in value]
    if isinstance(value, str):
        return value.replace("v2", "v3")
    return value


def build_audit(
    *,
    raw_summary: dict[str, Any],
    raw_gate: dict[str, Any],
    erratum_gate: dict[str, Any],
    mask_selection: dict[str, Any],
    v3_selection: dict[str, Any],
    mask_results_sha256: str,
    v3_results_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checks = {
        "v3_preference_gate_passed": erratum_gate.get("passed") is True,
        "v3_preference_gate_is_audited_erratum": erratum_gate.get("status")
        == "audited_arithmetic_erratum",
        "mask_selection_is_bound": mask_selection.get("method") == "mask"
        and mask_selection.get("results_sha256") == mask_results_sha256,
        "v3_selection_is_bound": v3_selection.get("method") == "v3"
        and v3_selection.get("results_sha256") == v3_results_sha256,
        "same_frozen_event_identity": mask_selection.get("event_set_hash")
        == v3_selection.get("event_set_hash")
        == raw_summary.get("event_set_hash")
        and mask_selection.get("event_file_sha256")
        == v3_selection.get("event_file_sha256")
        == raw_summary.get("event_file_sha256"),
        "selection_scoring_succeeded": int(mask_selection.get("scoring_failures", -1))
        == 0
        and int(v3_selection.get("scoring_failures", -1)) == 0,
        "raw_evaluator_binds_mask": raw_summary.get("mask_results_sha256")
        == mask_results_sha256,
        "raw_evaluator_binds_method_b_to_v3": raw_summary.get("v2_results_sha256")
        == v3_results_sha256,
        "reference_free_continuation": raw_summary.get(
            "continuation_input_excludes_evaluator_and_reference"
        )
        is True,
        "no_reference_suffix_or_test_access": raw_summary.get("reference_suffix_used")
        is False
        and raw_summary.get("test_split_access") is False,
    }
    relabeled_summary = _rename_v2(raw_summary)
    relabeled_summary.update(
        {
            "stage": "route_a_seed42_v3_expanded_appworld_bounded",
            "method_a": "mask",
            "method_b": "v3",
            "method_b_was_legacy_v2_cli_slot": True,
            "identity_checks": checks,
            "scope": (
                "controlled-state bounded-horizon AppWorld development evaluation; "
                "not fully autonomous task success and not fresh confirmatory evidence"
            ),
        }
    )
    relabeled_gate = _rename_v2(raw_gate)
    passed = bool(raw_gate.get("passed")) and all(checks.values())
    relabeled_gate.update(
        {
            "passed": passed,
            "stage": "route_a_seed42_v3_expanded_appworld_bounded_gate",
            "method_a": "mask",
            "method_b": "v3",
            "identity_checks": checks,
            "scope": relabeled_summary["scope"],
            "next_step": (
                "freeze a fresh confirmatory task partition and seeds"
                if passed
                else "do not expand seeds; inspect bounded-horizon development failures"
            ),
        }
    )
    return relabeled_summary, relabeled_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-summary", type=Path, required=True)
    parser.add_argument("--raw-gate", type=Path, required=True)
    parser.add_argument("--erratum-gate", type=Path, required=True)
    parser.add_argument("--mask-selection-summary", type=Path, required=True)
    parser.add_argument("--v3-selection-summary", type=Path, required=True)
    parser.add_argument("--mask-results", type=Path, required=True)
    parser.add_argument("--v3-results", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    args = parser.parse_args()

    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    summary, gate = build_audit(
        raw_summary=load(args.raw_summary),
        raw_gate=load(args.raw_gate),
        erratum_gate=load(args.erratum_gate),
        mask_selection=load(args.mask_selection_summary),
        v3_selection=load(args.v3_selection_summary),
        mask_results_sha256=file_sha256(args.mask_results),
        v3_results_sha256=file_sha256(args.v3_results),
    )
    write_json(args.summary_output, summary)
    write_json(args.gate_output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
