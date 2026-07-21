#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from rescuecredit.edit_credit import (
    fold_role,
    select_rescue_constrained_threshold,
    summarize_selection,
)
from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash
from scripts.freeze_toolsandbox_editcredit_protocol import STATUS


PROTECTED_SCORE_FIELDS = {
    "decision",
    "replay_valid",
    "branch_a",
    "branch_b",
    "target",
    "correct",
}


def _task_improvement_fraction(edit_rows, full_rows) -> float:
    edit_by_task: dict[str, list[bool]] = defaultdict(list)
    full_by_task: dict[str, list[bool]] = defaultdict(list)
    for row in edit_rows:
        edit_by_task[str(row["task_id_hash"])].append(bool(row["correct"]))
    for row in full_rows:
        full_by_task[str(row["task_id_hash"])].append(bool(row["correct"]))
    tasks = sorted(set(edit_by_task) & set(full_by_task))
    return sum(
        sum(edit_by_task[task]) / len(edit_by_task[task])
        > sum(full_by_task[task]) / len(full_by_task[task])
        for task in tasks
    ) / max(1, len(tasks))


def _decode_threshold(value: Any) -> float:
    return -math.inf if value is None else float(value)


def _same_threshold(left: float, right: float) -> bool:
    if math.isinf(left) or math.isinf(right):
        return left == right
    return abs(left - right) <= 1e-12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--data-gate", type=Path, required=True)
    parser.add_argument("--gradient-sanity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != STATUS:
        raise ValueError("invalid EditCredit protocol")
    train_rows = read_jsonl(args.train_file)
    truth = {str(row["event_id"]): row for row in train_rows}
    assignment = {str(key): int(value) for key, value in protocol["task_fold_assignment"].items()}
    folds = int(protocol["config"]["folds"])
    manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    data_gate = json.loads(args.data_gate.read_text(encoding="utf-8"))
    gradient_sanity = json.loads(args.gradient_sanity.read_text(encoding="utf-8"))

    integrity = {
        "protocol_status": True,
        "source_identity": bool(protocol.get("source_sha256"))
        and all(
            Path(path).is_file() and file_sha256(Path(path)) == expected
            for path, expected in protocol.get("source_sha256", {}).items()
        ),
        "frozen_bank_identity": file_sha256(args.train_file) == protocol.get("train_sha256")
        and event_set_hash(train_rows) == protocol.get("train_event_set_hash")
        and len(truth) == len(train_rows) == int(protocol["events"]),
        "v44_lineage_bound": file_sha256(args.data_manifest) == protocol.get("data_manifest_sha256")
        and file_sha256(args.data_gate) == protocol.get("data_gate_sha256")
        and manifest.get("status") == "frozen"
        and manifest.get("passed") is True
        and data_gate.get("passed") is True
        and manifest.get("train_sha256") == file_sha256(args.train_file),
        "gradient_sanity_bound": file_sha256(args.gradient_sanity)
        == protocol.get("gradient_sanity_sha256")
        and gradient_sanity.get("passed") is True
        and all(gradient_sanity.get("checks", {}).values()),
        "exact_replay_labels_only": all(
            row.get("replay_valid") is True
            and row.get("decision") in {"rescue_preference", "reverse_preference"}
            for row in train_rows
        ),
        "cross_task_group_isolation": True,
        "run_and_eval_bound": True,
        "public_scores_outcome_free": True,
        "score_derivations_recomputed": True,
        "calibration_threshold_recomputed": True,
        "prediction_identity": True,
        "source_identity_absent_for_editcredit": True,
    }
    method_test_scores: dict[str, list[dict[str, Any]]] = {
        "full_action": [],
        "editcredit": [],
    }
    edit_thresholds: dict[int, float] = {}
    presentation_side_aucs: list[float] = []

    for test_fold in range(folds):
        role_groups = {
            role: {
                str(row["task_id_hash"])
                for row in train_rows
                if fold_role(
                    row,
                    assignment=assignment,
                    test_fold=test_fold,
                    folds=folds,
                )
                == role
            }
            for role in ("train", "calibration", "test")
        }
        if (
            role_groups["train"] & role_groups["calibration"]
            or role_groups["train"] & role_groups["test"]
            or role_groups["calibration"] & role_groups["test"]
        ):
            integrity["cross_task_group_isolation"] = False

        expected_train = [
            row
            for row in train_rows
            if fold_role(
                row,
                assignment=assignment,
                test_fold=test_fold,
                folds=folds,
            )
            == "train"
        ]
        expected_scored_ids = {
            str(row["event_id"])
            for row in train_rows
            if fold_role(
                row,
                assignment=assignment,
                test_fold=test_fold,
                folds=folds,
            )
            in {"calibration", "test"}
        }

        for method in method_test_scores:
            directory = args.root / method / f"fold{test_fold}"
            run_path = directory / "run_summary.json"
            eval_path = directory / "eval" / "eval_summary.json"
            scores_path = directory / "eval" / "scores.public.jsonl"
            joined_path = directory / "eval" / "predictions.joined.jsonl"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            summary = json.loads(eval_path.read_text(encoding="utf-8"))
            if not (
                run.get("status") == summary.get("status") == "completed"
                and run.get("method") == summary.get("method") == method
                and int(run.get("fold", -1)) == int(summary.get("fold", -1)) == test_fold
                and run.get("protocol_lock_sha256") == summary.get("protocol_lock_sha256") == file_sha256(args.protocol_lock)
                and run.get("adapter_sha256") == summary.get("adapter_sha256")
                and run.get("adapter_sha256") == directory_sha256(Path(run["adapter"]))
                and Path(run["adapter"]).resolve() == (directory / "adapter").resolve()
                and run.get("base_model_sha256") == protocol.get("base_model_sha256")
                and run.get("train_file_sha256") == protocol.get("train_sha256")
                and all(run.get(key) == value for key, value in protocol["config"].items())
                and int(run.get("presentations", -1))
                == int(protocol["config"]["epochs"])
                * int(protocol["config"]["presentations_per_epoch"])
                and summary.get("run_summary_sha256") == file_sha256(run_path)
                and summary.get("public_scores_sha256") == file_sha256(scores_path)
                and summary.get("predictions_sha256") == file_sha256(joined_path)
            ):
                integrity["run_and_eval_bound"] = False
            if run.get("train_event_set_hash") != event_set_hash(expected_train):
                integrity["run_and_eval_bound"] = False
            if set(run.get("train_task_group_ids", [])) != role_groups["train"]:
                integrity["cross_task_group_isolation"] = False
            if method == "editcredit":
                if run.get("source_identity_in_model_input") is not False:
                    integrity["source_identity_absent_for_editcredit"] = False
                presentation_side_aucs.append(float(run["presentation_side_label_auc"]))

            scores = read_jsonl(scores_path)
            score_ids = [str(row["event_id"]) for row in scores]
            if len(score_ids) != len(set(score_ids)) or set(score_ids) != expected_scored_ids:
                integrity["prediction_identity"] = False
            if any(PROTECTED_SCORE_FIELDS & set(row) for row in scores):
                integrity["public_scores_outcome_free"] = False
            joined = []
            for score in scores:
                event_id = str(score["event_id"])
                bound = truth.get(event_id)
                if bound is None:
                    integrity["prediction_identity"] = False
                    continue
                expected_role = fold_role(
                    bound,
                    assignment=assignment,
                    test_fold=test_fold,
                    folds=folds,
                )
                if (
                    int(score.get("fold", -1)) != test_fold
                    or score.get("role") != expected_role
                    or str(score.get("task_id_hash")) != str(bound["task_id_hash"])
                ):
                    integrity["prediction_identity"] = False
                try:
                    margin_original = float(score["margin_original_order"])
                    margin_swapped = float(score["margin_swapped_order"])
                    reported_margin = float(score["margin_b_over_a"])
                except (KeyError, TypeError, ValueError):
                    integrity["score_derivations_recomputed"] = False
                    continue
                recomputed_margin = 0.5 * (margin_original + margin_swapped)
                recomputed_consistency = (margin_original >= 0.0) == (
                    margin_swapped >= 0.0
                )
                if not (
                    math.isfinite(margin_original)
                    and math.isfinite(margin_swapped)
                    and math.isfinite(reported_margin)
                    and abs(reported_margin - recomputed_margin) <= 1e-12
                    and score.get("swap_consistent") is recomputed_consistency
                ):
                    integrity["score_derivations_recomputed"] = False
                joined.append(
                    {
                        **score,
                        "margin_b_over_a": recomputed_margin,
                        "swap_consistent": recomputed_consistency,
                        "decision": str(bound["decision"]),
                    }
                )
            calibration = [row for row in joined if row["role"] == "calibration"]
            test = [row for row in joined if row["role"] == "test"]
            if method == "editcredit":
                choice = select_rescue_constrained_threshold(
                    calibration,
                    rescue_delta=float(protocol["config"]["rescue_delta"]),
                )
                recomputed_threshold = choice.threshold
                reported_threshold = _decode_threshold(summary.get("selected_threshold"))
                reported_constraint = summary.get("calibration_constraint") or {}
                if not (
                    _same_threshold(recomputed_threshold, reported_threshold)
                    and abs(float(reported_constraint.get("rescue_drop", -1)) - choice.rescue_drop) <= 1e-12
                    and abs(float(reported_constraint.get("reverse_recall", -1)) - choice.reverse_recall) <= 1e-12
                    and int(reported_constraint.get("route_to_a", -1)) == choice.route_to_a
                    and reported_constraint.get("feasible") is choice.feasible
                ):
                    integrity["calibration_threshold_recomputed"] = False
                edit_thresholds[test_fold] = recomputed_threshold
            method_test_scores[method].extend(test)

    expected_ids = set(truth)
    for rows in method_test_scores.values():
        ids = [str(row["event_id"]) for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            integrity["prediction_identity"] = False

    full = summarize_selection(method_test_scores["full_action"], threshold=0.0)
    edit_rows: list[dict[str, Any]] = []
    for test_fold in range(folds):
        fold_rows = [
            row
            for row in method_test_scores["editcredit"]
            if int(row["fold"]) == test_fold
        ]
        edit_rows.extend(
            summarize_selection(fold_rows, threshold=edit_thresholds[test_fold])["rows"]
        )
    rescue = [row for row in edit_rows if row["decision"] == "rescue_preference"]
    reverse = [row for row in edit_rows if row["decision"] == "reverse_preference"]
    edit = {
        "events": len(edit_rows),
        "accuracy": sum(row["correct"] for row in edit_rows) / max(1, len(edit_rows)),
        "rescue_accuracy": sum(row["correct"] for row in rescue) / max(1, len(rescue)),
        "reverse_recall": sum(row["correct"] for row in reverse) / max(1, len(reverse)),
    }
    edit["balanced_accuracy"] = 0.5 * (edit["rescue_accuracy"] + edit["reverse_recall"])
    mask = {
        "events": len(expected_ids),
        "rescue_accuracy": 1.0,
        "reverse_recall": 0.0,
        "balanced_accuracy": 0.5,
    }
    rescue_drop = mask["rescue_accuracy"] - edit["rescue_accuracy"]
    reverse_gain = edit["reverse_recall"]
    balanced_gain = edit["balanced_accuracy"] - full["balanced_accuracy"]
    task_fraction = _task_improvement_fraction(edit_rows, full["rows"])
    swap_consistency = sum(
        row["swap_consistent"] for row in method_test_scores["editcredit"]
    ) / len(method_test_scores["editcredit"])
    side_deviation = max(
        (abs(value - 0.5) for value in presentation_side_aucs), default=1.0
    )
    thresholds = protocol["gate_thresholds"]
    checks = {
        "rescue_noninferiority": rescue_drop <= thresholds["max_rescue_drop"] + 1e-12,
        "reverse_recall_gain": reverse_gain >= thresholds["min_reverse_recall_gain_over_mask"],
        "balanced_accuracy_gain": balanced_gain >= thresholds["min_balanced_accuracy_gain_over_full_action"],
        "task_macro_improvement": task_fraction >= thresholds["min_task_macro_improvement_fraction"],
        "presentation_side_balance": side_deviation <= thresholds["max_presentation_side_auc_deviation"] + 1e-12,
        "swap_consistency": swap_consistency >= thresholds["min_swap_consistency"],
    }
    passed = all(integrity.values()) and all(checks.values())
    output_rows = sorted(edit_rows, key=lambda row: str(row["event_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_output = args.output.with_name("oof_predictions.jsonl")
    write_jsonl(prediction_output, output_rows)
    gate = {
        "passed": passed,
        "feasibility_passed": passed,
        "stage": "toolsandbox_editcredit_seed42_gate",
        "integrity_checks": integrity,
        "outcome_checks": checks,
        "thresholds": thresholds,
        "observed": {
            "mask_default_b": mask,
            "full_action": {key: value for key, value in full.items() if key != "rows"},
            "editcredit_constrained": edit,
            "rescue_drop": rescue_drop,
            "reverse_recall_gain_over_mask": reverse_gain,
            "balanced_accuracy_gain_over_full_action": balanced_gain,
            "task_macro_improvement_fraction": task_fraction,
            "max_presentation_side_auc_deviation": side_deviation,
            "swap_consistency": swap_consistency,
        },
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "oof_predictions_sha256": file_sha256(prediction_output),
        "claim_boundary": protocol["claim_boundary"],
        "next_step": "expand to seeds 43/44" if passed else "stop EditCredit before additional seeds or fresh rollout",
    }
    write_json(args.output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
