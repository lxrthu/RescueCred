#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_preference import preference_kind
from scripts.train_route_a_preference import validity_first_epoch_order
from scripts.freeze_route_a_v3_expanded_protocol import (
    EXPECTED_TRAIN_SHA256,
    EXPECTED_VALIDATION_SHA256,
)


EXPECTED_CONFIG = {
    "method": "v31",
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

GATE_THRESHOLDS = {
    "min_validity_first_events": 20,
    "min_both_valid_causal_events": 5,
    "min_missing_required_accuracy": 0.90,
    "min_both_valid_accuracy_improvement": 0.05,
    "require_validity_first_noninferiority": True,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--mask-eval", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = args.data_root / "data"
    manifest_path = data / "manifest.json"
    gate_path = args.data_root / "data_gate.json"
    train_path = data / "train.jsonl"
    validation_path = data / "validation.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    mask_run = json.loads(args.mask_run.read_text(encoding="utf-8"))
    mask_eval = json.loads(args.mask_eval.read_text(encoding="utf-8"))
    model_sha = directory_sha256(args.model)
    checks = {
        "data_gate_passed": data_gate.get("passed") is True,
        "data_manifest_frozen": manifest.get("status") == "frozen",
        "exact_train_identity": file_sha256(train_path) == EXPECTED_TRAIN_SHA256,
        "exact_validation_identity": file_sha256(validation_path)
        == EXPECTED_VALIDATION_SHA256,
        "mask_is_frozen_baseline": mask_run.get("method") == "mask"
        and mask_run.get("presentations_per_epoch") == 255
        and mask_run.get("active_event_presentations") == 765,
        "mask_eval_bound": mask_eval.get("method") == "mask"
        and mask_eval.get("run_summary_sha256") == file_sha256(args.mask_run)
        and mask_eval.get("adapter_sha256") == mask_run.get("adapter_sha256"),
        "base_model_bound": mask_run.get("base_model_sha256") == model_sha
        and mask_eval.get("base_model_sha256") == model_sha,
        "no_dev_or_test_training": manifest.get("scope")
        == "train-only AppWorld causal preference data; no dev or test access",
    }
    if not all(checks.values()):
        raise RuntimeError(f"V3.1 protocol preflight failed: {checks}")

    source_paths = [
        Path("rescuecredit/route_a_preference.py"),
        Path("scripts/train_route_a_preference.py"),
        Path("scripts/evaluate_route_a_preference.py"),
        Path("scripts/freeze_route_a_v31_protocol.py"),
        Path("scripts/check_route_a_v31_gate.py"),
        Path("scripts/cloud/run_route_a_v31_validity_seed42.sh"),
    ]
    train_rows = read_jsonl(train_path)
    expected_presented_decisions: Counter[str] = Counter()
    for epoch in range(EXPECTED_CONFIG["epochs"]):
        for row in validity_first_epoch_order(
            train_rows,
            EXPECTED_CONFIG["seed"],
            epoch,
            len(train_rows),
        ):
            expected_presented_decisions[preference_kind(row, "v31")] += 1
    protocol = {
        "status": "frozen_before_v31_outcomes",
        "stage": "route_a_seed42_v31_validity_first",
        "checks": checks,
        "config": EXPECTED_CONFIG,
        "gate_thresholds": GATE_THRESHOLDS,
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(validation_path),
        "mask_run_sha256": file_sha256(args.mask_run),
        "mask_eval_sha256": file_sha256(args.mask_eval),
        "mask_adapter_sha256": mask_run["adapter_sha256"],
        "base_model_sha256": model_sha,
        "source_sha256": {str(path): file_sha256(path) for path in source_paths},
        "expected_presented_decisions": dict(
            sorted(expected_presented_decisions.items())
        ),
        "routing_rule": {
            "a_invalid_b_valid": "b_over_a",
            "a_valid_b_invalid": "a_over_b",
            "both_valid": "shadow_delta_direction",
            "unknown_or_both_invalid": "skip",
        },
        "sampling": "natural ratio after validity gate; no forced 50/50 causal balance",
        "reference_boundary": (
            "public schema validity plus frozen train shadow credit only; no dev/test"
        ),
    }
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
