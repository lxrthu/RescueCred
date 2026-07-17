#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256


MIN_CAUSAL_EVENTS = 15
MIN_ACCURACY_IMPROVEMENT = 0.10
EXPECTED_PRESENTATIONS_PER_EPOCH = 255
EXPECTED_ACTIVE_PRESENTATIONS = 765
EXPECTED_DECISIONS = {"rescue_preference": 384, "reverse_preference": 381}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--v3-run", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mask = json.loads(args.mask.read_text(encoding="utf-8"))
    mask_run = json.loads(args.mask_run.read_text(encoding="utf-8"))
    v3 = json.loads(args.v3.read_text(encoding="utf-8"))
    v3_run = json.loads(args.v3_run.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    improvement = float(v3["causal_accuracy"]) - float(mask["causal_accuracy"])
    base_sha = directory_sha256(Path(v3_run["model"]))
    checks = {
        "same_validation_split": mask["validation_file_sha256"]
        == v3["validation_file_sha256"]
        == protocol["validation_sha256"],
        "enough_causal_events": int(v3["causal_events"]) >= MIN_CAUSAL_EVENTS,
        "accuracy_improvement": improvement >= MIN_ACCURACY_IMPROVEMENT,
        "positive_signed_margin": float(v3["mean_signed_causal_margin"]) > 0,
        "reverse_accuracy_improvement": float(v3["reverse_accuracy"])
        > float(mask["reverse_accuracy"]),
        "frozen_thresholds": protocol.get("gate_thresholds")
        == {
            "min_causal_events": MIN_CAUSAL_EVENTS,
            "min_accuracy_improvement": MIN_ACCURACY_IMPROVEMENT,
            "require_positive_mean_signed_margin": True,
            "require_reverse_accuracy_improvement": True,
        },
        "matched_presentation_budget": mask_run.get("presentations_per_epoch")
        == v3_run.get("presentations_per_epoch")
        == EXPECTED_PRESENTATIONS_PER_EPOCH
        and mask_run.get("active_event_presentations")
        == v3_run.get("active_event_presentations")
        == EXPECTED_ACTIVE_PRESENTATIONS,
        "v3_balanced_causal_presentations": v3_run.get("presented_decisions")
        == EXPECTED_DECISIONS,
        "v3_excludes_zero_delta": v3_run.get(
            "zero_delta_rows_excluded_from_causal_loss"
        )
        is True,
        "methods_bound": mask.get("method") == mask_run.get("method") == "mask"
        and v3.get("method") == v3_run.get("method") == "v3",
        "mask_artifacts_bound": mask.get("run_summary_sha256")
        == file_sha256(args.mask_run)
        and mask.get("adapter_sha256") == mask_run.get("adapter_sha256")
        == protocol.get("mask_adapter_sha256"),
        "v3_artifacts_bound": v3.get("run_summary_sha256")
        == file_sha256(args.v3_run)
        and v3.get("adapter_sha256") == v3_run.get("adapter_sha256"),
        "protocol_bound": v3_run.get("protocol_lock_sha256")
        == file_sha256(args.protocol_lock)
        and protocol.get("status") == "frozen_before_v3_outcomes",
        "base_model_bound": base_sha == protocol.get("base_model_sha256")
        == mask.get("base_model_sha256")
        == v3.get("base_model_sha256"),
    }
    passed = all(checks.values())
    result = {
        "passed": passed,
        "stage": "route_a_seed42_v3_expanded_preference_gate",
        "checks": checks,
        "validation_causal_events": v3["causal_events"],
        "mask_causal_accuracy": mask["causal_accuracy"],
        "v3_causal_accuracy": v3["causal_accuracy"],
        "accuracy_improvement": improvement,
        "mask_rescue_accuracy": mask["rescue_accuracy"],
        "v3_rescue_accuracy": v3["rescue_accuracy"],
        "mask_reverse_accuracy": mask["reverse_accuracy"],
        "v3_reverse_accuracy": v3["reverse_accuracy"],
        "v3_mean_signed_causal_margin": v3["mean_signed_causal_margin"],
        "scope": "held-out train-task preference gate; not AppWorld task success",
        "next_step": (
            "run paired controlled-state AppWorld dev evaluation"
            if passed
            else "stop before AppWorld dev evaluation and inspect V3 learning"
        ),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
