#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_router import deterministic_group_folds
from rescuecredit.toolsandbox_selective_router import (
    PROBE_METHODS,
    apply_platt_scaler,
    average_precision,
    calibration_metrics,
    choose_conservative_threshold,
    fit_platt_scaler,
    fit_probe,
    probe_probabilities,
    reverse_target,
    roc_auc,
)
from scripts.freeze_toolsandbox_v6_protocol import CONFIG, PROTOCOL_STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=PROBE_METHODS, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch

    started = time.time()
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("config") != CONFIG:
        raise ValueError("invalid V6 diagnostic protocol")
    if file_sha256(args.feature_cache) != protocol.get("feature_cache_sha256"):
        raise ValueError("V6 feature cache identity mismatch")
    if file_sha256(args.feature_manifest) != protocol.get("feature_manifest_sha256"):
        raise ValueError("V6 feature manifest identity mismatch")
    if manifest.get("private_branch_outcomes_cached") is not False:
        raise ValueError("V6 feature cache contains private outcomes")

    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    feature_key = "control_features" if args.method == "margin_probe" else "semantic_features"
    features = cache[feature_key].float()
    labels = torch.tensor(
        [reverse_target(str(value)) for value in cache["decisions"]],
        dtype=torch.float32,
    )
    groups = [str(value) for value in cache["task_ids"]]
    folds = deterministic_group_folds(
        groups, folds=CONFIG["group_folds"], seed=CONFIG["seed"]
    )
    oof_raw = torch.zeros(len(labels), dtype=torch.float32)
    all_indices = set(range(len(labels)))
    fold_audit = []
    for fold_index, validation_indices in enumerate(folds):
        training_indices = sorted(all_indices - set(validation_indices))
        train_groups = {groups[index] for index in training_indices}
        validation_groups = {groups[index] for index in validation_indices}
        if train_groups & validation_groups:
            raise RuntimeError("task leakage across V6 folds")
        fitted = fit_probe(
            features[training_indices],
            labels[training_indices],
            method=args.method,
            seed=CONFIG["seed"] + fold_index * 1009,
            steps=CONFIG["head_steps"],
            learning_rate=CONFIG["head_learning_rate"],
            weight_decay=CONFIG["head_weight_decay"],
            hidden_dim=CONFIG["hidden_dim"],
        )
        oof_raw[validation_indices] = probe_probabilities(
            features[validation_indices], fitted
        )
        fold_audit.append(
            {
                "fold": fold_index,
                "training_events": len(training_indices),
                "validation_events": len(validation_indices),
                "training_tasks": len(train_groups),
                "validation_tasks": len(validation_groups),
                "task_overlap": 0,
            }
        )

    calibration = fit_platt_scaler(
        oof_raw,
        labels,
        steps=CONFIG["calibration_steps"],
        learning_rate=CONFIG["calibration_learning_rate"],
    )
    oof_calibrated = apply_platt_scaler(oof_raw, calibration)
    label_list = [int(value) for value in labels]
    raw_list = [float(value) for value in oof_raw]
    calibrated_list = [float(value) for value in oof_calibrated]
    cross_task_roc = roc_auc(label_list, raw_list)
    cross_task_pr = average_precision(label_list, raw_list)
    threshold_audit = choose_conservative_threshold(
        label_list,
        calibrated_list,
        CONFIG["threshold_candidates"],
        rescue_delta=CONFIG["rescue_delta"],
    )
    final = fit_probe(
        features,
        labels,
        method=args.method,
        seed=CONFIG["seed"] + 99991,
        steps=CONFIG["head_steps"],
        learning_rate=CONFIG["head_learning_rate"],
        weight_decay=CONFIG["head_weight_decay"],
        hidden_dim=CONFIG["hidden_dim"],
    )
    checkpoint = {
        key: value for key, value in final.items() if key != "train_probabilities"
    }
    checkpoint.update(
        {
            "calibration": calibration,
            "threshold": float(threshold_audit["selected"]["threshold"]),
            "feature_key": feature_key,
            "protocol_lock_sha256": file_sha256(args.protocol_lock),
            "feature_cache_sha256": file_sha256(args.feature_cache),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "probe.pt"
    torch.save(checkpoint, checkpoint_path)
    prevalence = sum(label_list) / len(label_list)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_v6_cross_task_reverse_diagnostic",
        "method": args.method,
        "events": len(labels),
        "tasks": len(set(groups)),
        "reverse_events": sum(label_list),
        "rescue_events": len(label_list) - sum(label_list),
        "reverse_prevalence": prevalence,
        "group_folds": CONFIG["group_folds"],
        "fold_audit": fold_audit,
        "cross_task_roc_auc": cross_task_roc,
        "cross_task_pr_auc": cross_task_pr,
        "cross_task_pr_auc_lift": cross_task_pr - prevalence,
        "raw_calibration": calibration_metrics(label_list, raw_list),
        "calibrated_calibration": calibration_metrics(label_list, calibrated_list),
        "platt_calibration": calibration,
        "selected_threshold": threshold_audit["selected"]["threshold"],
        "oof_selective_metrics": threshold_audit["selected"],
        "threshold_audit": threshold_audit["candidates"],
        "rescue_delta": CONFIG["rescue_delta"],
        "final_train_loss": final["loss"],
        "probe_sha256": file_sha256(checkpoint_path),
        "feature_cache_sha256": file_sha256(args.feature_cache),
        "feature_manifest_sha256": file_sha256(args.feature_manifest),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "base_and_mask_parameters_updated": False,
        "private_outcomes_used_as_features": False,
        "scope": "task-grouped out-of-fold diagnostic; labels are used only for probe training, calibration, and threshold selection",
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
