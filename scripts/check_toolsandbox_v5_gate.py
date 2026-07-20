#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_router import summarize_router_predictions
from scripts.freeze_toolsandbox_v5_protocol import PROTOCOL_STATUS, THRESHOLDS


TOL = 1e-12


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_exact(summary: dict, rows: list[dict]) -> bool:
    expected = summarize_router_predictions(rows)
    for key, value in expected.items():
        actual = summary.get(key)
        if isinstance(value, float):
            if actual is None or abs(float(actual) - value) > TOL:
                return False
        elif actual != value:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "mask-eval",
        "control-eval",
        "v5-eval",
        "mask-results",
        "control-results",
        "v5-results",
        "control-run",
        "v5-run",
        "control-router",
        "v5-router",
        "feature-cache",
        "feature-manifest",
        "scoring-summary",
        "protocol-lock",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mask, control, v5 = (
        load(args.mask_eval),
        load(args.control_eval),
        load(args.v5_eval),
    )
    control_run, v5_run = load(args.control_run), load(args.v5_run)
    scoring, protocol = load(args.scoring_summary), load(args.protocol_lock)
    mask_rows = read_jsonl(args.mask_results)
    control_rows = read_jsonl(args.control_results)
    v5_rows = read_jsonl(args.v5_results)
    mask_by_id = {str(row["event_id"]): row for row in mask_rows}
    control_by_id = {str(row["event_id"]): row for row in control_rows}
    v5_by_id = {str(row["event_id"]): row for row in v5_rows}
    exact_ids = (
        set(mask_by_id) == set(control_by_id) == set(v5_by_id)
        and len(mask_by_id) == len(mask_rows)
        and len(control_by_id) == len(control_rows)
        and len(v5_by_id) == len(v5_rows)
    )
    disagreements = [
        event_id
        for event_id in sorted(mask_by_id)
        if exact_ids
        and mask_by_id[event_id]["selected"] != v5_by_id[event_id]["selected"]
    ]
    wins = sum(
        v5_by_id[event_id]["causal_correct"]
        and not mask_by_id[event_id]["causal_correct"]
        for event_id in disagreements
    )
    losses = sum(
        mask_by_id[event_id]["causal_correct"]
        and not v5_by_id[event_id]["causal_correct"]
        for event_id in disagreements
    )
    source_ok = bool(protocol.get("source_sha256")) and all(
        Path(path).is_file() and file_sha256(Path(path)) == digest
        for path, digest in protocol.get("source_sha256", {}).items()
    )
    integrity = {
        "protocol_frozen": protocol.get("status") == PROTOCOL_STATUS
        and protocol.get("thresholds") == THRESHOLDS,
        "source_identity": source_ok,
        "router_runs_bound": control_run.get("protocol_lock_sha256")
        == v5_run.get("protocol_lock_sha256")
        == file_sha256(args.protocol_lock),
        "router_files_bound": control_run.get("router_sha256")
        == file_sha256(args.control_router)
        and v5_run.get("router_sha256") == file_sha256(args.v5_router),
        "scoring_router_files_bound": scoring.get("control_router_sha256")
        == file_sha256(args.control_router)
        and scoring.get("v5_router_sha256") == file_sha256(args.v5_router),
        "common_feature_cache_bound": control_run.get("feature_cache_sha256")
        == v5_run.get("feature_cache_sha256")
        == file_sha256(args.feature_cache)
        and control_run.get("feature_manifest_sha256")
        == v5_run.get("feature_manifest_sha256")
        == file_sha256(args.feature_manifest),
        "router_method_roles": control_run.get("method") == "margin_control"
        and v5_run.get("method") == "causal_router_v5"
        and control_run.get("group_folds")
        == v5_run.get("group_folds")
        == protocol["config"]["group_folds"],
        "frozen_model_unchanged": control_run.get("base_and_mask_parameters_updated")
        is False
        and v5_run.get("base_and_mask_parameters_updated") is False,
        "public_only_scoring": scoring.get("private_outcomes_read") is False
        and mask.get("public_only_model_scoring") is True
        and control.get("public_only_model_scoring") is True
        and v5.get("public_only_model_scoring") is True
        and mask.get("offline_outcomes_joined_after_scoring") is True
        and control.get("offline_outcomes_joined_after_scoring") is True
        and v5.get("offline_outcomes_joined_after_scoring") is True,
        "scoring_protocol_bound": scoring.get("protocol_lock_sha256")
        == mask.get("protocol_lock_sha256")
        == control.get("protocol_lock_sha256")
        == v5.get("protocol_lock_sha256")
        == file_sha256(args.protocol_lock),
        "prediction_files_bound": scoring.get("prediction_sha256", {}).get("mask")
        == mask.get("predictions_sha256")
        and scoring.get("prediction_sha256", {}).get("margin_control")
        == control.get("predictions_sha256")
        and scoring.get("prediction_sha256", {}).get("causal_router_v5")
        == v5.get("predictions_sha256"),
        "result_files_bound": mask.get("results_sha256")
        == file_sha256(args.mask_results)
        and control.get("results_sha256") == file_sha256(args.control_results)
        and v5.get("results_sha256") == file_sha256(args.v5_results),
        "same_development_events": exact_ids
        and mask.get("event_set_hash")
        == control.get("event_set_hash")
        == v5.get("event_set_hash")
        and mask.get("evaluation_role")
        == control.get("evaluation_role")
        == v5.get("evaluation_role")
        == "development",
        "raw_metrics_recomputed": summary_exact(mask, mask_rows)
        and summary_exact(control, control_rows)
        and summary_exact(v5, v5_rows),
    }
    terminal_gain = float(v5["mean_selected_terminal_similarity"]) - float(
        mask["mean_selected_terminal_similarity"]
    )
    progress_gain = float(v5["mean_selected_progress_auc"]) - float(
        mask["mean_selected_progress_auc"]
    )
    outcomes = {
        "enough_events": int(v5["valid_events"]) >= THRESHOLDS["min_events"],
        "enough_router_flips": len(disagreements) >= THRESHOLDS["min_router_flips"],
        "train_oof_improvement": float(v5_run["oof_accuracy_improvement"]) > TOL,
        "overall_beats_mask": float(v5["causal_accuracy"])
        > float(mask["causal_accuracy"]) + TOL,
        "overall_beats_margin_control": float(v5["causal_accuracy"])
        > float(control["causal_accuracy"]) + TOL,
        "rescue_noninferiority": float(v5["rescue_accuracy"])
        >= float(mask["rescue_accuracy"]) - TOL,
        "reverse_improvement": float(v5["reverse_accuracy"])
        > float(mask["reverse_accuracy"]) + TOL,
        "wins_over_losses": wins > losses,
        "terminal_noninferiority": terminal_gain >= -TOL,
        "progress_noninferiority": progress_gain >= -TOL,
    }
    passed = all(integrity.values()) and all(outcomes.values())
    gate = {
        "passed": passed,
        "stage": "toolsandbox_v5_independent_router_development_gate_seed42",
        "integrity_checks": integrity,
        "outcome_checks": outcomes,
        "thresholds": THRESHOLDS,
        "events": v5["valid_events"],
        "router_flips": len(disagreements),
        "router_wins": wins,
        "router_losses": losses,
        "mask_accuracy": mask["causal_accuracy"],
        "margin_control_accuracy": control["causal_accuracy"],
        "v5_accuracy": v5["causal_accuracy"],
        "v5_vs_mask": float(v5["causal_accuracy"]) - float(mask["causal_accuracy"]),
        "v5_vs_margin_control": float(v5["causal_accuracy"])
        - float(control["causal_accuracy"]),
        "mask_rescue_accuracy": mask["rescue_accuracy"],
        "v5_rescue_accuracy": v5["rescue_accuracy"],
        "mask_reverse_accuracy": mask["reverse_accuracy"],
        "v5_reverse_accuracy": v5["reverse_accuracy"],
        "terminal_similarity_improvement": terminal_gain,
        "progress_auc_improvement": progress_gain,
        "scope": protocol["scope"],
        "next_step": (
            "freeze a new ToolSandbox scenario profile before confirmation"
            if passed
            else "stop or revise the independent router before fresh confirmation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
