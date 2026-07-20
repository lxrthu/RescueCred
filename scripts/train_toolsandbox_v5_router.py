#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_router import (
    ROUTER_METHODS,
    choose_flip_threshold,
    deterministic_group_folds,
    fit_logistic_head,
    router_probabilities,
)
from scripts.freeze_toolsandbox_v5_protocol import CONFIG, PROTOCOL_STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=ROUTER_METHODS, required=True)
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
        raise ValueError("invalid V5 router protocol")
    if manifest.get("feature_file_sha256") != file_sha256(args.feature_cache):
        raise ValueError("V5 feature cache identity mismatch")
    if manifest.get("protocol_lock_sha256") != file_sha256(args.protocol_lock):
        raise ValueError("V5 features were not built under this protocol")
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    feature_key = (
        "control_features" if args.method == "margin_control" else "semantic_features"
    )
    features = cache[feature_key].float()
    labels = cache["labels"].float()
    groups = [str(value) for value in cache["task_ids"]]
    folds = deterministic_group_folds(
        groups, folds=CONFIG["group_folds"], seed=CONFIG["seed"]
    )
    oof = torch.zeros(len(labels), dtype=torch.float32)
    all_indices = set(range(len(labels)))
    for fold, validation_indices in enumerate(folds):
        training_indices = sorted(all_indices - set(validation_indices))
        fitted = fit_logistic_head(
            features[training_indices],
            labels[training_indices],
            seed=CONFIG["seed"] + fold * 1009,
            steps=CONFIG["head_steps"],
            learning_rate=CONFIG["head_learning_rate"],
            weight_decay=CONFIG["head_weight_decay"],
        )
        oof[validation_indices] = router_probabilities(
            features[validation_indices], fitted
        )
    threshold_audit = choose_flip_threshold(
        [float(value) for value in oof],
        [str(value) for value in cache["mask_selected"]],
        [str(value) for value in cache["decisions"]],
        CONFIG["threshold_candidates"],
        min_flips=CONFIG["min_oof_flips"],
    )
    final = fit_logistic_head(
        features,
        labels,
        seed=CONFIG["seed"] + 99991,
        steps=CONFIG["head_steps"],
        learning_rate=CONFIG["head_learning_rate"],
        weight_decay=CONFIG["head_weight_decay"],
    )
    checkpoint = {
        "method": args.method,
        "weight": final["weight"],
        "bias": final["bias"],
        "mean": final["mean"],
        "scale": final["scale"],
        "threshold": float(threshold_audit["selected"]["threshold"]),
        "input_dim": int(features.shape[1]),
        "feature_key": feature_key,
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "feature_cache_sha256": file_sha256(args.feature_cache),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "router.pt"
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_v5_independent_router_training",
        "method": args.method,
        "events": len(labels),
        "input_dim": int(features.shape[1]),
        "flip_labels": int(labels.sum()),
        "keep_labels": int(len(labels) - labels.sum()),
        "group_folds": CONFIG["group_folds"],
        "oof_baseline_accuracy": threshold_audit["baseline_accuracy"],
        "oof_router_accuracy": threshold_audit["selected"]["accuracy"],
        "oof_accuracy_improvement": threshold_audit["selected"]["accuracy"]
        - threshold_audit["baseline_accuracy"],
        "oof_flips": threshold_audit["selected"]["flips"],
        "oof_wins": threshold_audit["selected"]["wins"],
        "oof_losses": threshold_audit["selected"]["losses"],
        "selected_threshold": threshold_audit["selected"]["threshold"],
        "threshold_audit": threshold_audit["candidates"],
        "final_train_loss": final["loss"],
        "router_sha256": file_sha256(checkpoint_path),
        "feature_cache_sha256": file_sha256(args.feature_cache),
        "feature_manifest_sha256": file_sha256(args.feature_manifest),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "base_and_mask_parameters_updated": False,
        "scope": "train-task grouped out-of-fold router calibration; no development outcomes used",
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
