#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_preference import preference_kind
from scripts.freeze_route_a_v3_expanded_protocol import (
    EXPECTED_TRAIN_SHA256,
    EXPECTED_VALIDATION_SHA256,
)
from scripts.train_route_a_preference import validity_first_epoch_order


CONFIRMATORY_SEEDS = (43, 44, 45)
PREFERENCE_THRESHOLDS = {
    "min_validity_first_events": 20,
    "min_both_valid_causal_events": 5,
    "min_missing_required_accuracy": 0.90,
    "require_validity_first_noninferiority": True,
}
AGGREGATE_THRESHOLDS = {
    "minimum_positive_score_seeds": 3,
    "minimum_total_nonzero_events": 15,
    "require_positive_mean_score_improvement": True,
    "require_positive_mean_causal_accuracy_improvement": True,
    "require_aggregate_wins_over_losses": True,
}


def expected_config(seed: int) -> dict:
    return {
        "method": "v31",
        "seed": seed,
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


def expected_mask_config(seed: int) -> dict:
    config = expected_config(seed)
    config.update(
        {
            "method": "mask",
            "absolute_margin_coef": 0.0,
            "target_margin": 0.0,
        }
    )
    return config


SOURCE_PATHS = (
    "rescuecredit/route_a_preference.py",
    "scripts/train_route_a_preference.py",
    "scripts/evaluate_route_a_preference.py",
    "scripts/freeze_route_a_v31_confirm_protocol.py",
    "scripts/check_route_a_v31_confirm_preference.py",
    "scripts/freeze_route_a_v31_confirm_bounded_protocol.py",
    "scripts/analyze_route_a_v31_confirm.py",
    "scripts/cloud/run_route_a_v31_confirm_43_44_45.sh",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=CONFIRMATORY_SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = args.data_root / "data"
    train = data / "train.jsonl"
    validation = data / "validation.jsonl"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    data_gate = json.loads((args.data_root / "data_gate.json").read_text(encoding="utf-8"))
    checks = {
        "data_gate_passed": data_gate.get("passed") is True,
        "data_manifest_frozen": manifest.get("status") == "frozen",
        "train_identity": file_sha256(train) == EXPECTED_TRAIN_SHA256,
        "validation_identity": file_sha256(validation) == EXPECTED_VALIDATION_SHA256,
        "train_only": manifest.get("scope")
        == "train-only AppWorld causal preference data; no dev or test access",
        "source_files_present": all(Path(path).is_file() for path in SOURCE_PATHS),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V3.1 confirmatory preflight failed: {checks}")

    rows = read_jsonl(train)
    decisions: Counter[str] = Counter()
    for epoch in range(3):
        for row in validity_first_epoch_order(rows, args.seed, epoch, len(rows)):
            decisions[preference_kind(row, "v31")] += 1
    lock = {
        "status": "frozen_before_v31_outcomes",
        "stage": f"route_a_v31_confirm_seed{args.seed}",
        "seed": args.seed,
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "checks": checks,
        "config": expected_config(args.seed),
        "mask_config": expected_mask_config(args.seed),
        "preference_thresholds": PREFERENCE_THRESHOLDS,
        "aggregate_thresholds": AGGREGATE_THRESHOLDS,
        "train_sha256": file_sha256(train),
        "validation_sha256": file_sha256(validation),
        "base_model_sha256": directory_sha256(args.model),
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "expected_presented_decisions": dict(sorted(decisions.items())),
        "routing_rule": {
            "a_invalid_b_valid": "b_over_a",
            "a_valid_b_invalid": "a_over_b",
            "both_valid": "shadow_delta_direction",
            "unknown_or_both_invalid": "skip",
        },
        "reference_boundary": "frozen train shadow credit only; no dev/test outcomes",
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != lock:
            raise RuntimeError("existing V3.1 confirmatory lock differs")
    else:
        write_json(args.output, lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
