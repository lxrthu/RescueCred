#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_active_shadow import (
    acquisition_mask,
    active_decision_metrics,
    choose_active_route_threshold,
    minimum_zero_harm_calibration_size,
)
from rescuecredit.toolsandbox_router import deterministic_group_folds
from rescuecredit.toolsandbox_selective_router import (
    apply_platt_scaler,
    average_precision,
    fit_platt_scaler,
    fit_probe,
    probe_probabilities,
    roc_auc,
)
from scripts.freeze_toolsandbox_v7_protocol import CONFIG, PROTOCOL_STATUS


def _fit(features, labels, *, seed: int):
    return fit_probe(
        features,
        labels,
        method="semantic_probe",
        seed=seed,
        steps=CONFIG["head_steps"],
        learning_rate=CONFIG["head_learning_rate"],
        weight_decay=CONFIG["head_weight_decay"],
        hidden_dim=CONFIG["hidden_dim"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
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
        raise ValueError("invalid V7 training protocol")
    if manifest.get("feature_cache_sha256") != file_sha256(args.feature_cache):
        raise ValueError("V7 feature cache identity mismatch")
    if manifest.get("protocol_lock_sha256") != file_sha256(args.protocol_lock):
        raise ValueError("V7 feature cache protocol mismatch")
    if manifest.get("protected_fields_used") != []:
        raise ValueError("V7 feature cache used protected fields")

    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    active_features = cache["active_features"].float()
    static_features = cache["static_features"].float()
    labels = cache["labels"].float()
    groups = [str(value) for value in cache["task_ids"]]
    event_ids = [str(value) for value in cache["event_ids"]]
    outer_folds = deterministic_group_folds(
        groups, folds=CONFIG["group_folds"], seed=CONFIG["seed"]
    )
    all_indices = set(range(len(labels)))
    static_oof = torch.zeros(len(labels), dtype=torch.float32)
    active_raw_oof = torch.zeros(len(labels), dtype=torch.float32)
    active_calibrated_oof = torch.zeros(len(labels), dtype=torch.float32)
    probed_oof = [False] * len(labels)
    routed_oof = [False] * len(labels)
    fold_ids = [-1] * len(labels)
    acquisition_thresholds = [0.0] * len(labels)
    route_thresholds = [1.1] * len(labels)
    fold_audit = []
    for fold_index, evaluation_indices in enumerate(outer_folds):
        remaining_indices = sorted(all_indices - set(evaluation_indices))
        remaining_groups = [groups[index] for index in remaining_indices]
        inner_folds = deterministic_group_folds(
            remaining_groups,
            folds=CONFIG["group_folds"] - 1,
            seed=CONFIG["seed"] + fold_index * 101,
        )
        calibration_local = inner_folds[fold_index % len(inner_folds)]
        calibration_indices = [remaining_indices[index] for index in calibration_local]
        model_indices = sorted(set(remaining_indices) - set(calibration_indices))
        model_tasks = {groups[index] for index in model_indices}
        calibration_tasks = {groups[index] for index in calibration_indices}
        evaluation_tasks = {groups[index] for index in evaluation_indices}
        if (
            model_tasks & calibration_tasks
            or model_tasks & evaluation_tasks
            or calibration_tasks & evaluation_tasks
        ):
            raise RuntimeError("task leakage across nested V7 folds")
        static_head = _fit(
            static_features[model_indices],
            labels[model_indices],
            seed=CONFIG["seed"] + fold_index * 2003,
        )
        active_head = _fit(
            active_features[model_indices],
            labels[model_indices],
            seed=CONFIG["seed"] + fold_index * 2011,
        )
        static_calibration = probe_probabilities(
            static_features[calibration_indices], static_head
        )
        active_calibration_raw = probe_probabilities(
            active_features[calibration_indices], active_head
        )
        calibration = fit_platt_scaler(
            active_calibration_raw,
            labels[calibration_indices],
            steps=CONFIG["calibration_steps"],
            learning_rate=CONFIG["calibration_learning_rate"],
        )
        active_calibration = apply_platt_scaler(
            active_calibration_raw, calibration
        )
        calibration_static_list = [float(value) for value in static_calibration]
        calibration_active_list = [float(value) for value in active_calibration]
        calibration_labels = [int(labels[index]) for index in calibration_indices]
        calibration_event_ids = [event_ids[index] for index in calibration_indices]
        calibration_probed, acquisition_threshold = acquisition_mask(
            calibration_static_list,
            calibration_event_ids,
            max_probe_rate=CONFIG["max_probe_rate"],
        )
        route_audit = choose_active_route_threshold(
            calibration_labels,
            calibration_active_list,
            calibration_probed,
            CONFIG["route_threshold_candidates"],
            rescue_delta=CONFIG["rescue_delta"],
            alpha=CONFIG["risk_alpha"],
        )
        route_threshold = float(route_audit["selected"]["route_threshold"])
        static_evaluation = probe_probabilities(
            static_features[evaluation_indices], static_head
        )
        active_evaluation_raw = probe_probabilities(
            active_features[evaluation_indices], active_head
        )
        active_evaluation = apply_platt_scaler(active_evaluation_raw, calibration)
        static_oof[evaluation_indices] = static_evaluation
        active_raw_oof[evaluation_indices] = active_evaluation_raw
        active_calibrated_oof[evaluation_indices] = active_evaluation
        for local_index, global_index in enumerate(evaluation_indices):
            is_probed = float(static_evaluation[local_index]) >= acquisition_threshold
            route_to_a = bool(
                is_probed
                and float(active_evaluation[local_index]) >= route_threshold
            )
            probed_oof[global_index] = is_probed
            routed_oof[global_index] = route_to_a
            fold_ids[global_index] = fold_index
            acquisition_thresholds[global_index] = acquisition_threshold
            route_thresholds[global_index] = route_threshold
        fold_audit.append(
            {
                "fold": fold_index,
                "model_training_events": len(model_indices),
                "calibration_events": len(calibration_indices),
                "evaluation_events": len(evaluation_indices),
                "model_training_tasks": len(model_tasks),
                "calibration_tasks": len(calibration_tasks),
                "evaluation_tasks": len(evaluation_tasks),
                "task_overlap": 0,
                "acquisition_threshold": acquisition_threshold,
                "route_threshold": route_threshold,
            }
        )
    label_list = [int(value) for value in labels]
    static_list = [float(value) for value in static_oof]
    active_raw_list = [float(value) for value in active_raw_oof]
    active_list = [float(value) for value in active_calibrated_oof]
    pipeline_metrics = active_decision_metrics(
        label_list,
        probed_oof,
        routed_oof,
        alpha=CONFIG["risk_alpha"],
    )
    final_static = _fit(static_features, labels, seed=CONFIG["seed"] + 90001)
    final_active = _fit(active_features, labels, seed=CONFIG["seed"] + 90007)
    checkpoint = {
        "static_head": {
            key: value
            for key, value in final_static.items()
            if key != "train_probabilities"
        },
        "active_head": {
            key: value
            for key, value in final_active.items()
            if key != "train_probabilities"
        },
        "deployment_ready": False,
        "deployment_blocker": "requires a separate fixed-policy calibration set",
        "max_probe_rate": CONFIG["max_probe_rate"],
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "feature_cache_sha256": file_sha256(args.feature_cache),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "active_shadow.pt"
    torch.save(checkpoint, checkpoint_path)
    oof_rows = [
        {
            "event_id": event_id,
            "task_id_hash": group,
            "label": label,
            "static_score": static_score,
            "active_raw_score": active_raw_score,
            "active_score": active_score,
            "probed": is_probed,
            "routed_to_a": routed_to_a,
            "fold_id": fold_id,
            "acquisition_threshold": acquisition_threshold,
            "route_threshold": route_threshold,
        }
        for event_id, group, label, static_score, active_raw_score, active_score, is_probed, routed_to_a, fold_id, acquisition_threshold, route_threshold in zip(
            event_ids,
            groups,
            label_list,
            static_list,
            active_raw_list,
            active_list,
            probed_oof,
            routed_oof,
            fold_ids,
            acquisition_thresholds,
            route_thresholds,
            strict=True,
        )
    ]
    oof_path = args.output_dir / "oof_predictions.jsonl"
    write_jsonl(oof_path, oof_rows)
    active_auc = roc_auc(label_list, active_raw_list)
    static_auc = roc_auc(label_list, static_list)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_v7_active_shadow_nested_cross_task_oof",
        "events": len(labels),
        "tasks": len(set(groups)),
        "reverse_events": sum(label_list),
        "rescue_events": len(label_list) - sum(label_list),
        "group_folds": CONFIG["group_folds"],
        "fold_audit": fold_audit,
        "static_cross_task_roc_auc": static_auc,
        "active_cross_task_roc_auc": active_auc,
        "active_cross_task_pr_auc": average_precision(label_list, active_raw_list),
        "active_auc_gain_over_static": active_auc - static_auc,
        "pipeline_metrics": pipeline_metrics,
        "risk_certified_metrics": None,
        "risk_alpha": CONFIG["risk_alpha"],
        "rescue_delta": CONFIG["rescue_delta"],
        "minimum_zero_harm_rescue_calibration_events": minimum_zero_harm_calibration_size(
            CONFIG["rescue_delta"], alpha=CONFIG["risk_alpha"]
        ),
        "current_rescue_calibration_events": len(label_list) - sum(label_list),
        "formal_risk_certification_possible": False,
        "evaluation_protocol": "nested task cross-fitting: model train, calibration, and untouched evaluation tasks are disjoint in every outer fold",
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "oof_predictions_sha256": file_sha256(oof_path),
        "feature_cache_sha256": file_sha256(args.feature_cache),
        "feature_manifest_sha256": file_sha256(args.feature_manifest),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "base_and_mask_parameters_updated": False,
        "full_trajectory_outcomes_used_as_features": False,
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
