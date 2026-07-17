#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256
from scripts.freeze_route_a_v31_protocol import GATE_THRESHOLDS


def build_gate(
    mask: dict, mask_run: dict, v31: dict, v31_run: dict, protocol: dict
) -> dict:
    both_valid_improvement = float(v31["both_valid_causal_accuracy"]) - float(
        mask["both_valid_causal_accuracy"]
    )
    checks = {
        "same_validation_split": mask["validation_file_sha256"]
        == v31["validation_file_sha256"]
        == protocol["validation_sha256"],
        "enough_validity_first_events": int(v31["validity_first_events"])
        >= GATE_THRESHOLDS["min_validity_first_events"],
        "enough_both_valid_causal_events": int(v31["both_valid_causal_events"])
        >= GATE_THRESHOLDS["min_both_valid_causal_events"],
        "missing_required_is_safe": float(v31["missing_required_accuracy"])
        >= GATE_THRESHOLDS["min_missing_required_accuracy"]
        and float(v31["missing_required_accuracy"])
        >= float(mask["missing_required_accuracy"]),
        "validity_first_noninferior": float(v31["validity_first_accuracy"])
        >= float(mask["validity_first_accuracy"]),
        "both_valid_causal_improves": both_valid_improvement
        >= GATE_THRESHOLDS["min_both_valid_accuracy_improvement"],
        "thresholds_frozen": protocol.get("gate_thresholds") == GATE_THRESHOLDS,
        "matched_presentation_budget": mask_run.get("presentations_per_epoch")
        == v31_run.get("presentations_per_epoch")
        == 255
        and mask_run.get("active_event_presentations")
        == v31_run.get("active_event_presentations")
        == 765,
        "validity_first_training": v31_run.get("validity_first") is True,
        "no_forced_causal_balance": v31_run.get("causal_class_balanced") is False,
        "natural_ratio_presentations_exact": v31_run.get("presented_decisions")
        == protocol.get("expected_presented_decisions")
        and sum(v31_run.get("presented_decisions", {}).values()) == 765,
        "methods_bound": mask.get("method") == mask_run.get("method") == "mask"
        and v31.get("method") == v31_run.get("method") == "v31",
        "mask_artifacts_bound": mask.get("run_summary_sha256")
        == mask_run.get("_actual_run_summary_sha256")
        == protocol.get("mask_run_sha256")
        and mask.get("adapter_sha256") == mask_run.get("adapter_sha256")
        == protocol.get("mask_adapter_sha256"),
        "mask_eval_frozen": mask.get("_actual_eval_sha256")
        == protocol.get("mask_eval_sha256"),
        "v31_adapter_bound": v31.get("run_summary_sha256")
        == v31_run.get("run_summary_sha256")
        and v31.get("adapter_sha256") == v31_run.get("adapter_sha256"),
        "protocol_bound": v31_run.get("protocol_lock_sha256")
        == protocol.get("_actual_protocol_lock_sha256"),
        "source_identity_frozen": bool(protocol.get("source_identity_matches")),
        "base_model_bound": v31.get("base_model_sha256")
        == mask.get("base_model_sha256")
        == protocol.get("base_model_sha256"),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "stage": "route_a_seed42_v31_validity_first_gate",
        "checks": checks,
        "mask_validity_first_accuracy": mask["validity_first_accuracy"],
        "v31_validity_first_accuracy": v31["validity_first_accuracy"],
        "mask_missing_required_accuracy": mask["missing_required_accuracy"],
        "v31_missing_required_accuracy": v31["missing_required_accuracy"],
        "both_valid_causal_events": v31["both_valid_causal_events"],
        "mask_both_valid_causal_accuracy": mask["both_valid_causal_accuracy"],
        "v31_both_valid_causal_accuracy": v31["both_valid_causal_accuracy"],
        "both_valid_accuracy_improvement": both_valid_improvement,
        "scope": "held-out train-task validity-first preference gate; not task success",
        "next_step": (
            "build a both-actions-valid AppWorld development event set"
            if passed
            else "stop and inspect V3.1 validity-first learning"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--v31", type=Path, required=True)
    parser.add_argument("--v31-run", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    mask = load(args.mask)
    mask_run = load(args.mask_run)
    v31 = load(args.v31)
    v31_run = load(args.v31_run)
    protocol = load(args.protocol_lock)
    # Bind hashes without mutating the immutable training summary on disk.
    mask["_actual_eval_sha256"] = file_sha256(args.mask)
    mask_run["_actual_run_summary_sha256"] = file_sha256(args.mask_run)
    v31_run["run_summary_sha256"] = file_sha256(args.v31_run)
    protocol["_actual_protocol_lock_sha256"] = file_sha256(args.protocol_lock)
    protocol["source_identity_matches"] = bool(protocol.get("source_sha256")) and all(
        Path(path).is_file() and file_sha256(Path(path)) == expected
        for path, expected in protocol.get("source_sha256", {}).items()
    )
    result = build_gate(mask, mask_run, v31, v31_run, protocol)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
