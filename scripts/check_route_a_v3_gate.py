#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256


MIN_CAUSAL_EVENTS = 5
MIN_ACCURACY_IMPROVEMENT = 0.10
EXPECTED_CONFIG = {
    "method": "v3",
    "seed": 42,
    "epochs": 3,
    "learning_rate": 3e-6,
    "gradient_accumulation": 8,
    "max_length": 2048,
    "beta": 1.0,
    "max_causal_weight": 2.5,
    "v2_presentations_per_epoch": 0,
    "absolute_margin_coef": 1.0,
    "target_margin": 0.05,
    "lora_r": 16,
    "lora_alpha": 32,
    "fp32": True,
}


def build_gate(
    mask: dict,
    v3: dict,
    run: dict,
    protocol: dict,
    *,
    mask_run: dict,
    mask_run_sha256: str,
    mask_eval_sha256: str,
    run_summary_sha256: str,
    protocol_lock_sha256: str,
    base_model_sha256: str,
) -> dict:
    improvement = float(v3["causal_accuracy"]) - float(mask["causal_accuracy"])
    checks = {
        "same_validation_split": mask["validation_file_sha256"]
        == v3["validation_file_sha256"],
        "enough_causal_validation_events": int(v3["causal_events"])
        >= MIN_CAUSAL_EVENTS,
        "v3_improves_causal_accuracy": improvement
        >= MIN_ACCURACY_IMPROVEMENT,
        "v3_positive_mean_signed_margin": float(v3["mean_signed_causal_margin"])
        > 0,
        "v3_improves_reverse_accuracy": float(v3["reverse_accuracy"])
        > float(mask["reverse_accuracy"]),
        "exact_frozen_gate_thresholds": protocol.get("gate_thresholds")
        == {
            "min_causal_events": MIN_CAUSAL_EVENTS,
            "min_accuracy_improvement": MIN_ACCURACY_IMPROVEMENT,
            "require_positive_mean_signed_margin": True,
            "require_reverse_accuracy_improvement": True,
        },
        "exact_v3_training_config": all(
            run.get(key) == value for key, value in EXPECTED_CONFIG.items()
        ),
        "exact_matched_presentation_budget": run.get("presentations_per_epoch")
        == 86
        and run.get("active_event_presentations") == 258
        and run.get("presentation_budget_matches_mask") is True
        and run.get("zero_delta_rows_excluded_from_causal_loss") is True
        and run.get("presented_decisions")
        == {"rescue_preference": 129, "reverse_preference": 129},
        "method_labels_are_bound": mask.get("method") == "mask"
        and v3.get("method") == "v3"
        and run.get("method") == "v3",
        "frozen_data_identity": run.get("train_file_sha256")
        == protocol.get("train_sha256")
        and v3.get("validation_file_sha256")
        == protocol.get("validation_sha256"),
        "mask_identity_is_frozen": mask_eval_sha256
        == protocol.get("mask_eval_sha256")
        and mask_run_sha256 == protocol.get("mask_run_sha256")
        and mask_run.get("method") == "mask"
        and directory_sha256(Path(mask_run["adapter"]))
        == protocol.get("mask_adapter_sha256")
        and mask.get("adapter_sha256") == protocol.get("mask_adapter_sha256")
        and mask.get("run_summary_sha256") == mask_run_sha256,
        "base_model_identity_is_bound": base_model_sha256
        == protocol.get("base_model_sha256")
        and run.get("base_model_sha256") == base_model_sha256
        and v3.get("base_model_sha256") == base_model_sha256
        and mask.get("base_model_sha256") == base_model_sha256
        and directory_sha256(Path(mask_run["model"])) == base_model_sha256,
        "adapter_and_run_are_bound": v3.get("adapter_sha256")
        == run.get("adapter_sha256")
        and v3.get("run_summary_sha256") == run_summary_sha256,
        "protocol_lock_is_bound": run.get("protocol_lock_sha256")
        == protocol_lock_sha256,
        "protocol_was_frozen_before_outcomes": protocol.get("status")
        == "frozen_before_v3_outcomes",
    }
    return {
        "passed": all(checks.values()),
        "stage": "route_a_seed42_v3_absolute_margin_engineering_gate",
        "gate_thresholds": {
            "min_causal_events": MIN_CAUSAL_EVENTS,
            "min_accuracy_improvement": MIN_ACCURACY_IMPROVEMENT,
        },
        "checks": checks,
        "mask_causal_accuracy": mask["causal_accuracy"],
        "v3_causal_accuracy": v3["causal_accuracy"],
        "accuracy_improvement": improvement,
        "mask_reverse_accuracy": mask["reverse_accuracy"],
        "v3_reverse_accuracy": v3["reverse_accuracy"],
        "v3_rescue_accuracy": v3["rescue_accuracy"],
        "scope": "engineering gate only; does not establish AppWorld task success",
        "next_step": (
            "freeze V3 and regenerate paired AppWorld selections"
            if all(checks.values())
            else "stop before AppWorld dev evaluation; V3 absolute-margin objective failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mask = json.loads(args.mask.read_text(encoding="utf-8"))
    mask_run = json.loads(args.mask_run.read_text(encoding="utf-8"))
    v3 = json.loads(args.v3.read_text(encoding="utf-8"))
    run = json.loads(args.run_summary.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    gate = build_gate(
        mask,
        v3,
        run,
        protocol,
        mask_run=mask_run,
        mask_run_sha256=file_sha256(args.mask_run),
        mask_eval_sha256=file_sha256(args.mask),
        run_summary_sha256=file_sha256(args.run_summary),
        protocol_lock_sha256=file_sha256(args.protocol_lock),
        base_model_sha256=directory_sha256(Path(run["model"])),
    )
    args.output.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
