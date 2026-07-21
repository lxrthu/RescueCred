#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rescuecredit.edit_credit import fold_role, stratified_group_folds
from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash


STATUS = "frozen_before_editcredit_crossfit_training"
CONFIG = {
    "seed": 42,
    "folds": 5,
    "epochs": 3,
    "learning_rate": 3e-6,
    "gradient_accumulation": 8,
    "max_length": 2048,
    "beta": 1.0,
    "absolute_margin_coef": 1.0,
    "target_margin": 0.05,
    "reference_anchor_coef": 0.25,
    "presentations_per_epoch": 126,
    "lora_r": 16,
    "lora_alpha": 32,
    "fp32": True,
    "rescue_delta": 0.02,
}
METHODS = ("full_action", "editcredit")
EFFICIENCY_CONFIG = {
    "checkpoint_presentations": [0, 40, 80, 128, 256, 378],
    "gradient_sketch_buckets": 128,
    "gradient_bootstrap_replicates": 2000,
    "gradient_bootstrap_batch_size": 8,
    "max_gradient_noise_scale_ratio": 0.70,
    "max_minibatch_gradient_mse_ratio": 0.70,
    "min_baseline_adjusted_balanced_auc_gain": 0.05,
    "min_relative_balanced_auc_gain": 0.10,
    "max_presentations_to_target_ratio": 0.50,
    "require_final_balanced_noninferiority": True,
    "max_final_rescue_drop": 0.02,
}
SOURCE_PATHS = (
    "rescuecredit/edit_credit.py",
    "rescuecredit/frozen_bank.py",
    "rescuecredit/toolsandbox_preference.py",
    "scripts/train_route_a_preference.py",
    "scripts/audit_editcredit_gradients.py",
    "scripts/freeze_toolsandbox_editcredit_protocol.py",
    "scripts/train_toolsandbox_editcredit.py",
    "scripts/evaluate_toolsandbox_editcredit.py",
    "scripts/check_toolsandbox_editcredit_gate.py",
    "scripts/audit_toolsandbox_editcredit_gradients.py",
    "scripts/check_toolsandbox_editcredit_variance.py",
    "scripts/check_toolsandbox_editcredit_efficiency.py",
    "scripts/cloud/run_toolsandbox_editcredit_seed42.sh",
    "tests/test_edit_credit.py",
    "tests/test_editcredit_gate.py",
    "tests/test_editcredit_efficiency.py",
    "refine-logs/EDITCREDIT_EXPERIMENT_PLAN_20260721_151030.md",
    "refine-logs/EDITCREDIT_EXPERIMENT_PLAN_ERRATUM_20260721_154254.md",
    "refine-logs/EDITCREDIT_EFFICIENCY_PREREG_20260721_155518.md",
)


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["decision"]) for row in rows).items()))


def build_protocol(
    train_file: Path,
    model: Path,
    data_manifest_path: Path,
    data_gate_path: Path,
    gradient_sanity_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    data_gate = json.loads(data_gate_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("passed") is not True:
        raise ValueError("V4.4 data manifest is not frozen/passed")
    if data_gate.get("passed") is not True:
        raise ValueError("V4.4 data gate did not pass")
    if manifest.get("train_sha256") != file_sha256(train_file):
        raise ValueError("V4.4 manifest does not bind the train bank")
    if manifest.get("events") != 126 or data_gate.get("events") != 126:
        raise ValueError("V4.4 lineage does not contain 126 events")
    if manifest.get("official_branch_metrics_in_training_file") is not False or manifest.get("protected_outcomes_in_prompt") is not False:
        raise ValueError("V4.4 public/private boundary is not sealed")
    gradient_sanity = json.loads(gradient_sanity_path.read_text(encoding="utf-8"))
    if gradient_sanity.get("passed") is not True or not all(
        gradient_sanity.get("checks", {}).values()
    ):
        raise ValueError("EditCredit gradient ownership sanity did not pass")
    rows = read_jsonl(train_file)
    if len(rows) != 126:
        raise ValueError(f"expected frozen 126-pair bank, got {len(rows)}")
    if any(row.get("replay_valid") is not True for row in rows):
        raise ValueError("EditCredit requires exact replay-valid pairs")
    decisions = Counter(str(row.get("decision")) for row in rows)
    if decisions != Counter({"rescue_preference": 41, "reverse_preference": 85}):
        raise ValueError(f"unexpected frozen decision counts: {decisions}")
    groups = {str(row["task_id_hash"]) for row in rows}
    if len(groups) != 38:
        raise ValueError(f"expected 38 task groups, got {len(groups)}")
    assignment = stratified_group_folds(rows, folds=CONFIG["folds"], seed=CONFIG["seed"])
    split_audit: list[dict[str, Any]] = []
    for test_fold in range(CONFIG["folds"]):
        role_rows = {
            role: [
                row
                for row in rows
                if fold_role(
                    row,
                    assignment=assignment,
                    test_fold=test_fold,
                    folds=CONFIG["folds"],
                )
                == role
            ]
            for role in ("train", "calibration", "test")
        }
        role_groups = {
            role: {str(row["task_id_hash"]) for row in selected}
            for role, selected in role_rows.items()
        }
        if any(set(_counts(selected)) != {"rescue_preference", "reverse_preference"} for selected in role_rows.values()):
            raise RuntimeError(f"fold {test_fold} lacks a causal class: { {k: _counts(v) for k, v in role_rows.items()} }")
        if role_groups["train"] & role_groups["calibration"] or role_groups["train"] & role_groups["test"] or role_groups["calibration"] & role_groups["test"]:
            raise RuntimeError("task groups cross train/calibration/test roles")
        split_audit.append(
            {
                "fold": test_fold,
                "roles": {
                    role: {
                        "events": len(selected),
                        "groups": len(role_groups[role]),
                        "decisions": _counts(selected),
                        "event_ids": sorted(str(row["event_id"]) for row in selected),
                    }
                    for role, selected in role_rows.items()
                },
            }
        )
    missing_sources = [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"missing EditCredit sources: {missing_sources}")
    return {
        "status": STATUS,
        "stage": "toolsandbox_editcredit_cross_task_seed42",
        "methods": list(METHODS),
        "config": CONFIG,
        "efficiency_config": EFFICIENCY_CONFIG,
        "train_file": str(train_file),
        "train_sha256": file_sha256(train_file),
        "data_manifest": str(data_manifest_path),
        "data_manifest_sha256": file_sha256(data_manifest_path),
        "data_gate": str(data_gate_path),
        "data_gate_sha256": file_sha256(data_gate_path),
        "gradient_sanity": str(gradient_sanity_path),
        "gradient_sanity_sha256": file_sha256(gradient_sanity_path),
        "source_event_sha256": manifest.get("source_event_sha256"),
        "source_summary_sha256": manifest.get("source_summary_sha256"),
        "source_protocol_sha256": manifest.get("source_protocol_sha256"),
        "train_event_set_hash": event_set_hash(rows),
        "events": len(rows),
        "task_groups": len(groups),
        "decisions": dict(sorted(decisions.items())),
        "task_fold_assignment": dict(sorted(assignment.items())),
        "split_audit": split_audit,
        "base_model_sha256": directory_sha256(model),
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "outcomes_visible_to": "training direction and post-score evaluation only",
        "model_inputs": "public prompt, candidate pair, and one changed field",
        "gate_thresholds": {
            "max_rescue_drop": 0.02,
            "min_reverse_recall_gain_over_mask": 0.10,
            "min_balanced_accuracy_gain_over_full_action": 0.05,
            "min_task_macro_improvement_fraction": 0.50,
            "max_presentation_side_auc_deviation": 0.05,
            "min_swap_consistency": 0.95,
        },
        "claim_boundary": "cross-task feasibility on the frozen V4.4 pair bank; not fresh task confirmation or autonomous rollout evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--data-gate", type=Path, required=True)
    parser.add_argument("--gradient-sanity", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace protocol: {args.output}")
    protocol = build_protocol(
        args.train_file,
        args.model,
        args.data_manifest,
        args.data_gate,
        args.gradient_sanity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
