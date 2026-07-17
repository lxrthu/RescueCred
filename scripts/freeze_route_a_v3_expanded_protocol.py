#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256
from rescuecredit.logging import write_json


EXPECTED_TRAIN_SHA256 = "67119a7f5a6dbf0e74715f630276a908247385626b427000d273bdeec962a730"
EXPECTED_VALIDATION_SHA256 = "fb1bec44fa8ae7ff815d93db979dbc455196c1886aba061ba77ec2436963d3e5"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--mask-eval", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_dir = args.data_root / "data"
    manifest_path = data_dir / "manifest.json"
    gate_path = args.data_root / "data_gate.json"
    repair_path = args.data_root / "repair_manifest.json"
    train_path = data_dir / "train.jsonl"
    validation_path = data_dir / "validation.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    mask_run = json.loads(args.mask_run.read_text(encoding="utf-8"))
    mask_eval = json.loads(args.mask_eval.read_text(encoding="utf-8"))
    train_sha = file_sha256(train_path)
    validation_sha = file_sha256(validation_path)
    mask_adapter = Path(mask_run.get("adapter", ""))
    model_sha = directory_sha256(args.model)
    mask_adapter_sha = directory_sha256(mask_adapter)
    checks = {
        "data_gate_passed": data_gate.get("passed") is True,
        "repair_is_frozen": repair.get("status") == "frozen",
        "selection_is_outcome_independent": repair.get(
            "selection_rule_frozen", {}
        ).get("balance_rule_uses_delta_or_preference")
        is False,
        "task_cap_is_10": repair.get("selection_rule_frozen", {}).get(
            "max_events_per_task"
        )
        == 10,
        "data_manifest_is_frozen": manifest.get("status") == "frozen",
        "data_counts_are_exact": manifest.get("train", {}).get("events") == 255
        and manifest.get("validation", {}).get("events") == 58
        and manifest.get("validation", {}).get("nonzero_events") == 16,
        "data_hashes_match_manifest": manifest.get("train_sha256") == train_sha
        and manifest.get("validation_sha256") == validation_sha,
        "exact_frozen_train_hash": train_sha == EXPECTED_TRAIN_SHA256,
        "exact_frozen_validation_hash": validation_sha
        == EXPECTED_VALIDATION_SHA256,
        "mask_method": mask_run.get("method") == "mask",
        "mask_budget": mask_run.get("presentations_per_epoch") == 255
        and mask_run.get("active_event_presentations") == 765,
        "mask_eval_bound": mask_eval.get("method") == "mask"
        and mask_eval.get("adapter_sha256") == mask_adapter_sha
        and mask_eval.get("run_summary_sha256") == file_sha256(args.mask_run),
        "same_base_model": mask_run.get("model") == str(args.model)
        and mask_run.get("base_model_sha256") == model_sha
        and mask_eval.get("base_model_sha256") == model_sha,
    }
    if not all(checks.values()):
        raise RuntimeError(f"expanded V3 protocol preflight failed: {checks}")

    source_paths = [
        Path("rescuecredit/route_a_preference.py"),
        Path("scripts/train_route_a_preference.py"),
        Path("scripts/evaluate_route_a_preference.py"),
        Path("scripts/check_route_a_v3_expanded_gate.py"),
        Path("scripts/freeze_route_a_v3_expanded_protocol.py"),
        Path("scripts/cloud/run_route_a_v3_expanded_seed42.sh"),
    ]
    protocol = {
        "status": "frozen_before_v3_outcomes",
        "stage": "route_a_seed42_v3_expanded",
        "checks": checks,
        "config": EXPECTED_CONFIG,
        "gate_thresholds": {
            "min_causal_events": 15,
            "min_accuracy_improvement": 0.10,
            "require_positive_mean_signed_margin": True,
            "require_reverse_accuracy_improvement": True,
        },
        "train_sha256": train_sha,
        "validation_sha256": validation_sha,
        "data_manifest_sha256": file_sha256(manifest_path),
        "data_gate_sha256": file_sha256(gate_path),
        "repair_manifest_sha256": file_sha256(repair_path),
        "mask_run_sha256": file_sha256(args.mask_run),
        "mask_eval_sha256": file_sha256(args.mask_eval),
        "mask_adapter_sha256": mask_adapter_sha,
        "base_model_sha256": model_sha,
        "source_sha256": {str(path): file_sha256(path) for path in source_paths},
        "reference_boundary": (
            "frozen train shadow-credit labels only; no AppWorld dev/test or "
            "private audit enters training"
        ),
    }
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
