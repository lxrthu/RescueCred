#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256
from rescuecredit.logging import write_json


FIXED_CONFIG = {
    "seed": 42,
    "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    "max_updates": 10000,
    "budget_mode": "main",
    "main_interaction_budget": 1200,
    "total_interaction_budget": 50000,
    "group_size": 4,
    "max_new_tokens": 64,
    "max_shadow_steps": 12,
    "temperature": 0.7,
    "policy_epochs": 1,
    "learning_rate": 2e-6,
    "clip_eps": 0.2,
    "kl_coef": 0.02,
    "audit_probability": 1.0,
    "audit_warm_start_events": 0,
    "lambda_corr": 0.1,
    "lambda_causal": 0.1,
    "visible_curriculum_fraction": 0.0,
    "use_lora": True,
    "fp32": True,
    "diagnostic_full_shadow": True,
    "force_shadow_credit_for_rescue": True,
    "strict_main_budget": True,
    "world_size": 1,
    "save_every": 5,
    "eval_max_new_tokens": 64,
    "eval_harness_mode": "oracle",
    "eval_generation": "greedy_do_sample_false",
}
METHODS = ["naive_h_grpo", "mask_correction", "rescuecredit"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = args.data_dir / "manifest.json"
    train = args.data_dir / "train.jsonl"
    dev = args.data_dir / "dev.jsonl"
    checks = {
        "manifest_exists": manifest.is_file(),
        "train_exists": train.is_file(),
        "dev_exists": dev.is_file(),
        "model_exists": args.model.is_dir(),
        "train_dev_distinct": file_sha256(train) != file_sha256(dev),
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol preflight failed: {checks}")

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    source_paths = [
        Path("environments/__init__.py"),
        Path("environments/api_bank/adapter.py"),
        Path("environments/api_bank/__init__.py"),
        Path("environments/api_bank/harness.py"),
        Path("environments/api_bank/shadow.py"),
        Path("environments/api_bank/correction_generator.py"),
        Path("environments/api_bank/data.py"),
        Path("environments/api_bank/deployable.py"),
        Path("environments/api_bank/verifier.py"),
        Path("rescuecredit/audit.py"),
        Path("rescuecredit/__init__.py"),
        Path("rescuecredit/accounting.py"),
        Path("rescuecredit/correction_preference.py"),
        Path("rescuecredit/engine.py"),
        Path("rescuecredit/estimators.py"),
        Path("rescuecredit/frozen_bank.py"),
        Path("rescuecredit/training.py"),
        Path("rescuecredit/types.py"),
        Path("rescuecredit/logging.py"),
        Path("rescuecredit/visible_curriculum.py"),
        Path("rescuecredit/v2_preference.py"),
        Path("configs/accelerate_h200.yaml"),
        Path("scripts/run_train.py"),
        Path("scripts/run_eval.py"),
        Path("scripts/analyze_harness_blindness.py"),
        Path("scripts/freeze_harness_blindness_protocol.py"),
        Path("scripts/cloud/run_harness_blindness_seed42.sh"),
    ]
    lock = {
        "status": "frozen_before_training",
        "stage": "harness_credit_blindness_seed42",
        "scope": (
            "controlled mechanism experiment with an oracle evidence-correction "
            "Harness; not deployable-Harness evidence"
        ),
        "methods": METHODS,
        "config": FIXED_CONFIG,
        "checks": checks,
        "data": {
            "manifest_sha256": file_sha256(manifest),
            "train_sha256": file_sha256(train),
            "dev_sha256": file_sha256(dev),
            "train_split_hash": manifest_data["split_hashes"]["train"],
            "dev_split_hash": manifest_data["split_hashes"]["dev"],
        },
        "base_model": {
            "path": str(args.model),
            "directory_sha256": directory_sha256(args.model),
        },
        "gate": {
            "min_diagnostic_rescue_events": 5,
            "max_naive_negative_prefix_rate_on_rescues": 0.0,
            "require_positive_naive_dependence_gap": True,
            "require_rescue_off_success_gain": True,
            "require_rescue_first_attempt_gain": True,
            "require_rescue_intervention_rate_reduction": True,
        },
        "source_sha256": {str(path): file_sha256(path) for path in source_paths},
        "reference_boundary": (
            "Controlled oracle-assisted mechanism experiment. expected_action is "
            "available to the Oracle Harness and official checker. After an "
            "intervention, the corrected executed action, patch id, and visible tool "
            "receipt enter later policy history; the reference action is not inserted "
            "as an explicit pre-intervention policy label. This is not deployable-"
            "Harness evidence."
        ),
    }
    write_json(args.output, lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
