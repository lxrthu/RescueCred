#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256
from rescuecredit.logging import write_json
from scripts.freeze_route_a_v31_confirm_protocol import PREFERENCE_THRESHOLDS


CONFIG_FIELDS = (
    "method",
    "seed",
    "epochs",
    "learning_rate",
    "gradient_accumulation",
    "max_length",
    "beta",
    "max_causal_weight",
    "v2_presentations_per_epoch",
    "absolute_margin_coef",
    "target_margin",
    "lora_r",
    "lora_alpha",
    "fp32",
)


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
    mask, mask_run = load(args.mask), load(args.mask_run)
    v31, v31_run, lock = load(args.v31), load(args.v31_run), load(args.protocol_lock)
    seed = int(lock["seed"])
    improvement = float(v31["both_valid_causal_accuracy"]) - float(
        mask["both_valid_causal_accuracy"]
    )
    checks = {
        "seed_bound": mask_run.get("seed") == v31_run.get("seed") == seed,
        "same_validation": mask["validation_file_sha256"]
        == v31["validation_file_sha256"]
        == lock["validation_sha256"],
        "enough_validity_first": int(v31["validity_first_events"])
        >= PREFERENCE_THRESHOLDS["min_validity_first_events"],
        "enough_both_valid": int(v31["both_valid_causal_events"])
        >= PREFERENCE_THRESHOLDS["min_both_valid_causal_events"],
        "missing_required_safe": float(v31["missing_required_accuracy"])
        >= PREFERENCE_THRESHOLDS["min_missing_required_accuracy"],
        "validity_first_noninferior": float(v31["validity_first_accuracy"])
        >= float(mask["validity_first_accuracy"]),
        "matched_budget": mask_run.get("presentations_per_epoch")
        == v31_run.get("presentations_per_epoch")
        and mask_run.get("active_event_presentations")
        == v31_run.get("active_event_presentations"),
        "v31_sampling_bound": v31_run.get("validity_first") is True
        and v31_run.get("causal_class_balanced") is False
        and v31_run.get("presented_decisions")
        == lock.get("expected_presented_decisions"),
        "methods_bound": mask_run.get("method") == mask.get("method") == "mask"
        and v31_run.get("method") == v31.get("method") == "v31",
        "artifacts_bound": mask.get("run_summary_sha256") == file_sha256(args.mask_run)
        and v31.get("run_summary_sha256") == file_sha256(args.v31_run)
        and v31_run.get("protocol_lock_sha256") == file_sha256(args.protocol_lock),
        "thresholds_frozen": lock.get("preference_thresholds")
        == PREFERENCE_THRESHOLDS,
        "mask_config_frozen": {
            field: mask_run.get(field) for field in CONFIG_FIELDS
        }
        == lock.get("mask_config"),
        "v31_config_frozen": {
            field: v31_run.get(field) for field in CONFIG_FIELDS
        }
        == lock.get("config"),
        "train_and_model_identity": mask_run.get("train_file_sha256")
        == v31_run.get("train_file_sha256") == lock.get("train_sha256")
        and mask_run.get("base_model_sha256")
        == v31_run.get("base_model_sha256") == lock.get("base_model_sha256"),
    }
    result = {
        "passed": all(checks.values()),
        "stage": f"route_a_v31_confirm_preference_seed{seed}",
        "seed": seed,
        "checks": checks,
        "mask_validity_first_accuracy": mask["validity_first_accuracy"],
        "v31_validity_first_accuracy": v31["validity_first_accuracy"],
        "mask_both_valid_causal_accuracy": mask["both_valid_causal_accuracy"],
        "v31_both_valid_causal_accuracy": v31["both_valid_causal_accuracy"],
        "both_valid_accuracy_improvement": improvement,
        "scope": "confirmatory held-out train-task preference diagnostic",
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
