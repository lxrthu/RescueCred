#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256
from rescuecredit.logging import write_json


EXPECTED_TRAIN_SHA256 = "ae31f4af761ac66bec28132dece6f584c5a40ed81967fe2bea1d50776ea181d4"
EXPECTED_VALIDATION_SHA256 = "734c088f676949684cada1617231d5734b73f14911408e101d5f80a33aeb14e6"
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
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--mask-eval", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.data_dir / "manifest.json"
    train_path = args.data_dir / "train.jsonl"
    validation_path = args.data_dir / "validation.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mask_run = json.loads(args.mask_run.read_text(encoding="utf-8"))
    mask_eval = json.loads(args.mask_eval.read_text(encoding="utf-8"))
    mask_adapter = Path(mask_run.get("adapter", ""))
    mask_model = Path(mask_run.get("model", ""))
    base_model_sha = directory_sha256(args.model)
    mask_adapter_sha = directory_sha256(mask_adapter)
    train_sha = file_sha256(train_path)
    validation_sha = file_sha256(validation_path)
    checks = {
        "manifest_is_frozen": manifest.get("status") == "frozen",
        "seed_is_42": manifest.get("seed") == 42,
        "event_counts_are_exact": manifest.get("train_events") == 86
        and manifest.get("validation_events") == 23,
        "manifest_hashes_match_files": manifest.get("train_sha256") == train_sha
        and manifest.get("validation_sha256") == validation_sha,
        "exact_frozen_train_hash": train_sha == EXPECTED_TRAIN_SHA256,
        "exact_frozen_validation_hash": validation_sha
        == EXPECTED_VALIDATION_SHA256,
        "train_and_validation_are_distinct": train_sha != validation_sha,
        "mask_artifacts_exist": args.mask_run.is_file()
        and args.mask_eval.is_file()
        and mask_adapter.is_dir(),
        "mask_method_is_bound": mask_run.get("method") == "mask",
        "mask_eval_is_bound_to_run": mask_eval.get("method") == "mask"
        and mask_eval.get("adapter_sha256") == mask_adapter_sha
        and mask_eval.get("run_summary_sha256") == file_sha256(args.mask_run),
        "mask_and_v3_base_model_match": mask_model == args.model
        and directory_sha256(mask_model) == base_model_sha
        and mask_eval.get("base_model_sha256") == base_model_sha,
        "model_config_exists": (args.model / "config.json").is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V3 protocol preflight failed: {checks}")

    source_paths = [
        Path("rescuecredit/route_a_preference.py"),
        Path("scripts/train_route_a_preference.py"),
        Path("scripts/evaluate_route_a_preference.py"),
        Path("scripts/check_route_a_v3_gate.py"),
        Path("scripts/freeze_route_a_v3_protocol.py"),
        Path("scripts/cloud/run_route_a_v3_absolute_seed42.sh"),
    ]
    protocol = {
        "status": "frozen_before_v3_outcomes",
        "stage": "route_a_seed42_v3_absolute_margin",
        "checks": checks,
        "config": EXPECTED_CONFIG,
        "gate_thresholds": {
            "min_causal_events": 5,
            "min_accuracy_improvement": 0.10,
            "require_positive_mean_signed_margin": True,
            "require_reverse_accuracy_improvement": True,
        },
        "manifest_sha256": file_sha256(manifest_path),
        "train_sha256": train_sha,
        "validation_sha256": validation_sha,
        "mask_run_sha256": file_sha256(args.mask_run),
        "mask_eval_sha256": file_sha256(args.mask_eval),
        "mask_adapter_sha256": mask_adapter_sha,
        "base_model_sha256": base_model_sha,
        "base_model_config_sha256": file_sha256(args.model / "config.json"),
        "source_sha256": {
            str(path): file_sha256(path) for path in source_paths
        },
        "reference_boundary": (
            "train shadow-credit labels only; no dev/test or private audit access"
        ),
    }
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
