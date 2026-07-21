#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rescuecredit.edit_credit import (
    fold_role,
    select_rescue_constrained_threshold,
    summarize_selection,
    trapezoid_auc,
)
from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash
from scripts.check_toolsandbox_editcredit_gate import PROTECTED_SCORE_FIELDS
from scripts.freeze_toolsandbox_editcredit_protocol import STATUS


def _score_directory(root: Path, method: str, fold: int, presentations: int, final: int) -> Path:
    base = root / method / f"fold{fold}"
    return base / "eval" if presentations == final else base / "curve" / f"p{presentations:06d}"


def _same_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    if math.isinf(left_value) or math.isinf(right_value):
        return left_value == right_value
    return math.isfinite(left_value) and math.isfinite(right_value) and abs(left_value - right_value) <= tolerance


def _reported_threshold(value: Any) -> float:
    return -math.inf if value is None else float(value)


def _rebuild_scores(scores, truth, assignment, fold, folds, integrity):
    rebuilt = []
    for score in scores:
        event_id = str(score.get("event_id"))
        bound = truth.get(event_id)
        if bound is None or PROTECTED_SCORE_FIELDS & set(score):
            integrity["public_scores_outcome_free"] = False
            continue
        expected_role = fold_role(
            bound, assignment=assignment, test_fold=fold, folds=folds
        )
        if (
            expected_role not in {"calibration", "test"}
            or score.get("role") != expected_role
            or int(score.get("fold", -1)) != fold
            or str(score.get("task_id_hash")) != str(bound["task_id_hash"])
        ):
            integrity["fold_roles_rebuilt"] = False
        try:
            original = float(score["margin_original_order"])
            swapped = float(score["margin_swapped_order"])
            reported = float(score["margin_b_over_a"])
        except (KeyError, TypeError, ValueError):
            integrity["score_derivations_rebuilt"] = False
            continue
        mean = 0.5 * (original + swapped)
        consistency = (original >= 0.0) == (swapped >= 0.0)
        if not (
            all(math.isfinite(value) for value in (original, swapped, reported))
            and abs(mean - reported) <= 1e-12
            and score.get("swap_consistent") is consistency
        ):
            integrity["score_derivations_rebuilt"] = False
        rebuilt.append(
            {
                **score,
                "margin_b_over_a": mean,
                "swap_consistent": consistency,
                "decision": str(bound["decision"]),
            }
        )
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variance-gate", type=Path, required=True)
    parser.add_argument("--final-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != STATUS:
        raise ValueError("invalid EditCredit protocol")
    train_rows = read_jsonl(args.train_file)
    truth = {str(row["event_id"]): row for row in train_rows}
    assignment = {str(key): int(value) for key, value in protocol["task_fold_assignment"].items()}
    folds = int(protocol["config"]["folds"])
    checkpoints = [int(value) for value in protocol["efficiency_config"]["checkpoint_presentations"]]
    checkpoint_tags = {f"p{value:06d}" for value in checkpoints}
    final = checkpoints[-1]
    variance_gate = json.loads(args.variance_gate.read_text(encoding="utf-8"))
    final_gate = json.loads(args.final_gate.read_text(encoding="utf-8"))
    protocol_sha256 = file_sha256(args.protocol_lock)
    integrity = {
        "protocol_status": True,
        "train_bank_bound": file_sha256(args.train_file) == protocol.get("train_sha256")
        and event_set_hash(train_rows) == protocol.get("train_event_set_hash"),
        "source_identity": bool(protocol.get("source_sha256"))
        and all(
            Path(path).is_file() and file_sha256(Path(path)) == expected
            for path, expected in protocol.get("source_sha256", {}).items()
        ),
        "variance_gate_bound": variance_gate.get("protocol_lock_sha256")
        == protocol_sha256
        and all(variance_gate.get("integrity_checks", {}).values()),
        "final_gate_bound": final_gate.get("protocol_lock_sha256")
        == protocol_sha256
        and all(final_gate.get("integrity_checks", {}).values()),
        "run_and_manifest_bound": True,
        "all_checkpoints_bound": True,
        "checkpoint_score_identity": True,
        "public_scores_outcome_free": True,
        "fold_roles_rebuilt": True,
        "score_derivations_rebuilt": True,
        "calibration_rebuilt": True,
        "oof_identity": True,
        "final_metrics_match_primary_gate": True,
    }
    curves: dict[str, list[dict[str, Any]]] = {"full_action": [], "editcredit": []}
    for presentations in checkpoints:
        method_test_rows: dict[str, list[dict[str, Any]]] = {
            "full_action": [],
            "editcredit": [],
        }
        edit_thresholds = {}
        wall_times = {"full_action": [], "editcredit": []}
        for method in method_test_rows:
            for fold in range(folds):
                run_path = args.root / method / f"fold{fold}" / "run_summary.json"
                run = json.loads(run_path.read_text(encoding="utf-8"))
                manifest_path = Path(run.get("checkpoint_manifest", ""))
                manifest = (
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.is_file()
                    else {}
                )
                manifest_sha256 = file_sha256(manifest_path) if manifest_path.is_file() else None
                tag = f"p{presentations:06d}"
                checkpoint = run.get("checkpoints", {}).get(tag)
                directory = _score_directory(args.root, method, fold, presentations, final)
                summary_path = directory / "eval_summary.json"
                scores_path = directory / "scores.public.jsonl"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                expected_train = [
                    row
                    for row in train_rows
                    if fold_role(row, assignment=assignment, test_fold=fold, folds=folds)
                    == "train"
                ]
                expected_scored_ids = {
                    str(row["event_id"])
                    for row in train_rows
                    if fold_role(row, assignment=assignment, test_fold=fold, folds=folds)
                    in {"calibration", "test"}
                }
                checkpoint_times = [
                    float(run.get("checkpoints", {}).get(checkpoint_tag, {}).get("wall_time_sec", -1.0))
                    for checkpoint_tag in sorted(checkpoint_tags)
                ]
                if not (
                    run.get("status") == "completed"
                    and run.get("method") == method
                    and int(run.get("fold", -1)) == fold
                    and run.get("protocol_lock_sha256") == protocol_sha256
                    and run.get("base_model_sha256") == protocol.get("base_model_sha256")
                    and run.get("train_file_sha256") == protocol.get("train_sha256")
                    and run.get("train_event_set_hash") == event_set_hash(expected_train)
                    and all(run.get(key) == value for key, value in protocol["config"].items())
                    and int(run.get("presentations", -1)) == final
                    and set(run.get("checkpoints", {})) == checkpoint_tags
                    and checkpoint_times == sorted(checkpoint_times)
                    and checkpoint_times[0] == 0.0
                    and checkpoint_times[-1] <= float(run.get("wall_time_sec", -1.0))
                    and run.get("checkpoint_manifest_sha256") == manifest_sha256
                    and manifest.get("status") == "completed"
                    and manifest.get("method") == method
                    and int(manifest.get("fold", -1)) == fold
                    and manifest.get("protocol_lock_sha256") == protocol_sha256
                    and manifest.get("checkpoints") == run.get("checkpoints")
                ):
                    integrity["run_and_manifest_bound"] = False
                expected_adapter = None
                if presentations > 0:
                    expected_adapter = (
                        args.root / method / f"fold{fold}" / "adapter"
                        if presentations == final
                        else args.root
                        / method
                        / f"fold{fold}"
                        / "checkpoints"
                        / tag
                        / "adapter"
                    )
                if not (
                    isinstance(checkpoint, dict)
                    and int(checkpoint.get("presentations", -1)) == presentations
                    and int(checkpoint.get("optimizer_steps", -1))
                    == math.ceil(
                        presentations
                        / int(protocol["config"]["gradient_accumulation"])
                    )
                    and checkpoint.get("base_model_sha256") == protocol.get("base_model_sha256")
                    and (
                        (expected_adapter is None and checkpoint.get("adapter") is None)
                        or (
                            expected_adapter is not None
                            and Path(checkpoint.get("adapter", "")).resolve()
                            == expected_adapter.resolve()
                        )
                    )
                    and summary.get("status") == "completed"
                    and summary.get("method") == method
                    and int(summary.get("fold", -1)) == fold
                    and int(summary.get("checkpoint_presentations", -1)) == presentations
                    and summary.get("protocol_lock_sha256") == protocol_sha256
                    and summary.get("run_summary_sha256") == file_sha256(run_path)
                    and summary.get("public_scores_sha256") == file_sha256(scores_path)
                    and summary.get("adapter_sha256") == checkpoint.get("adapter_sha256")
                    and (
                        checkpoint.get("adapter") is None
                        or (
                            Path(checkpoint["adapter"]).is_dir()
                            and directory_sha256(Path(checkpoint["adapter"]))
                            == checkpoint.get("adapter_sha256")
                        )
                    )
                ):
                    integrity["all_checkpoints_bound"] = False
                wall_times[method].append(
                    float(checkpoint.get("wall_time_sec", 0.0))
                    if isinstance(checkpoint, dict)
                    else 0.0
                )
                scores = read_jsonl(scores_path)
                score_ids = [str(row.get("event_id")) for row in scores]
                if len(score_ids) != len(set(score_ids)) or set(score_ids) != expected_scored_ids:
                    integrity["checkpoint_score_identity"] = False
                rebuilt = _rebuild_scores(scores, truth, assignment, fold, folds, integrity)
                calibration = [row for row in rebuilt if row["role"] == "calibration"]
                test = [row for row in rebuilt if row["role"] == "test"]
                if method == "editcredit":
                    choice = select_rescue_constrained_threshold(
                        calibration,
                        rescue_delta=float(protocol["config"]["rescue_delta"]),
                    )
                    reported_threshold = _reported_threshold(summary.get("selected_threshold"))
                    reported_constraint = summary.get("calibration_constraint") or {}
                    if not (
                        _same_number(choice.threshold, reported_threshold)
                        and _same_number(
                            choice.rescue_drop, reported_constraint.get("rescue_drop")
                        )
                        and _same_number(
                            choice.reverse_recall,
                            reported_constraint.get("reverse_recall"),
                        )
                        and int(reported_constraint.get("route_to_a", -1))
                        == choice.route_to_a
                        and reported_constraint.get("feasible") is choice.feasible
                    ):
                        integrity["calibration_rebuilt"] = False
                    edit_thresholds[fold] = choice.threshold
                elif summary.get("selected_threshold") is not None or summary.get(
                    "calibration_constraint"
                ) is not None:
                    integrity["calibration_rebuilt"] = False
                recomputed_test = summarize_selection(
                    test,
                    threshold=edit_thresholds[fold] if method == "editcredit" else 0.0,
                )
                reported_test = summary.get("test", {})
                if any(
                    not _same_number(recomputed_test.get(metric), reported_test.get(metric))
                    for metric in (
                        "events",
                        "accuracy",
                        "rescue_accuracy",
                        "reverse_recall",
                        "balanced_accuracy",
                        "route_to_a",
                    )
                ):
                    integrity["calibration_rebuilt"] = False
                method_test_rows[method].extend(test)
        for method, rows in method_test_rows.items():
            ids = [str(row["event_id"]) for row in rows]
            if len(ids) != len(set(ids)) or set(ids) != set(truth):
                integrity["oof_identity"] = False
            if method == "full_action":
                selected = summarize_selection(rows, threshold=0.0)
            else:
                selected_rows = []
                for fold in range(folds):
                    selected_rows.extend(
                        summarize_selection(
                            [row for row in rows if int(row["fold"]) == fold],
                            threshold=edit_thresholds[fold],
                        )["rows"]
                    )
                rescue = [row for row in selected_rows if row["decision"] == "rescue_preference"]
                reverse = [row for row in selected_rows if row["decision"] == "reverse_preference"]
                rescue_accuracy = sum(row["correct"] for row in rescue) / len(rescue)
                reverse_recall = sum(row["correct"] for row in reverse) / len(reverse)
                selected = {
                    "events": len(selected_rows),
                    "accuracy": sum(row["correct"] for row in selected_rows) / len(selected_rows),
                    "rescue_accuracy": rescue_accuracy,
                    "reverse_recall": reverse_recall,
                    "balanced_accuracy": 0.5 * (rescue_accuracy + reverse_recall),
                }
            curves[method].append(
                {
                    "presentations": presentations,
                    **{key: value for key, value in selected.items() if key != "rows"},
                    "mean_training_wall_time_sec": sum(wall_times[method]) / len(wall_times[method]),
                }
            )

    full_auc = trapezoid_auc(
        [(row["presentations"], row["balanced_accuracy"]) for row in curves["full_action"]]
    )
    edit_auc = trapezoid_auc(
        [(row["presentations"], row["balanced_accuracy"]) for row in curves["editcredit"]]
    )
    relative_auc_gain = (edit_auc - full_auc) / max(abs(full_auc), 1e-12)
    full_final = curves["full_action"][-1]
    edit_final = curves["editcredit"][-1]
    target = full_final["balanced_accuracy"]
    reached = [
        row["presentations"]
        for row in curves["editcredit"]
        if row["balanced_accuracy"] >= target - 1e-12
        and 1.0 - row["rescue_accuracy"]
        <= float(protocol["efficiency_config"]["max_final_rescue_drop"]) + 1e-12
    ]
    earliest = min(reached) if reached else None
    presentation_ratio = earliest / final if earliest is not None else None
    variance_noise_ratio = float(variance_gate["observed"]["gradient_noise_scale_ratio"])
    variance_mse_ratio = float(variance_gate["observed"]["minibatch_gradient_mse_ratio"])
    full_baseline = curves["full_action"][0]["balanced_accuracy"]
    edit_baseline = curves["editcredit"][0]["balanced_accuracy"]
    full_adjusted_auc = trapezoid_auc(
        [
            (row["presentations"], row["balanced_accuracy"] - full_baseline)
            for row in curves["full_action"]
        ]
    )
    edit_adjusted_auc = trapezoid_auc(
        [
            (row["presentations"], row["balanced_accuracy"] - edit_baseline)
            for row in curves["editcredit"]
        ]
    )
    baseline_adjusted_auc_gain = edit_adjusted_auc - full_adjusted_auc
    checks = {
        "primary_final_gate_passed": final_gate.get("passed") is True,
        "gradient_noise_scale": variance_noise_ratio
        <= float(protocol["efficiency_config"]["max_gradient_noise_scale_ratio"]),
        "minibatch_gradient_mse": variance_mse_ratio
        <= float(protocol["efficiency_config"]["max_minibatch_gradient_mse_ratio"]),
        "relative_balanced_auc_gain": relative_auc_gain
        >= float(protocol["efficiency_config"]["min_relative_balanced_auc_gain"]),
        "baseline_adjusted_balanced_auc_gain": baseline_adjusted_auc_gain
        >= float(
            protocol["efficiency_config"]["min_baseline_adjusted_balanced_auc_gain"]
        ),
        "presentations_to_target": presentation_ratio is not None
        and presentation_ratio
        <= float(protocol["efficiency_config"]["max_presentations_to_target_ratio"]),
        "final_balanced_noninferiority": edit_final["balanced_accuracy"]
        >= full_final["balanced_accuracy"] - 1e-12,
        "final_rescue_noninferiority": 1.0 - edit_final["rescue_accuracy"]
        <= float(protocol["efficiency_config"]["max_final_rescue_drop"]) + 1e-12,
    }
    primary_observed = final_gate.get("observed", {})
    for key, value in (
        ("full_action", full_final),
        ("editcredit_constrained", edit_final),
    ):
        primary = primary_observed.get(key, {})
        if any(
            abs(float(primary.get(metric, float("nan"))) - float(value[metric])) > 1e-12
            for metric in ("rescue_accuracy", "reverse_recall", "balanced_accuracy")
        ):
            integrity["final_metrics_match_primary_gate"] = False
    passed = all(integrity.values()) and all(checks.values())
    result = {
        "passed": passed,
        "seed42_efficiency_feasibility_supported": passed,
        "paper_facing_efficiency_claim_supported": False,
        "stage": "toolsandbox_editcredit_efficiency_gate_seed42",
        "integrity_checks": integrity,
        "outcome_checks": checks,
        "curves": curves,
        "observed": {
            "full_action_balanced_auc": full_auc,
            "editcredit_balanced_auc": edit_auc,
            "relative_balanced_auc_gain": relative_auc_gain,
            "full_action_p0_balanced_accuracy": full_baseline,
            "editcredit_p0_balanced_accuracy": edit_baseline,
            "p0_balanced_accuracy_advantage": edit_baseline - full_baseline,
            "full_action_baseline_adjusted_balanced_auc": full_adjusted_auc,
            "editcredit_baseline_adjusted_balanced_auc": edit_adjusted_auc,
            "baseline_adjusted_balanced_auc_gain": baseline_adjusted_auc_gain,
            "full_action_final_target": target,
            "editcredit_presentations_to_target": earliest,
            "presentations_to_target_ratio": presentation_ratio,
            "gradient_noise_scale_ratio": variance_noise_ratio,
            "minibatch_gradient_mse_ratio": variance_mse_ratio,
            "final_full_action": full_final,
            "final_editcredit": edit_final,
        },
        "thresholds": protocol["efficiency_config"],
        "variance_gate_sha256": file_sha256(args.variance_gate),
        "final_gate_sha256": file_sha256(args.final_gate),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "claim_boundary": "seed-42 frozen-bank feasibility only; system sample efficiency requires absolute and p0-adjusted held-out convergence, lower method-specific gradient noise, and the primary final gate",
        "next_step": "expand to seeds 43/44" if passed else "do not claim EditCredit variance or convergence improvement",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
