#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash
from rescuecredit.toolsandbox_router import (
    desired_candidate,
    summarize_router_predictions,
)
from scripts.freeze_toolsandbox_v5_protocol import PROTOCOL_STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("mask", "margin_control", "causal_router_v5"),
        required=True,
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--scoring-summary", type=Path, required=True)
    parser.add_argument("--public-events", type=Path, required=True)
    parser.add_argument("--private-outcomes", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument(
        "--evaluation-role", choices=("development", "posthoc"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    scoring = json.loads(args.scoring_summary.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("invalid V5 evaluation protocol")
    if scoring.get("protocol_lock_sha256") != file_sha256(args.protocol_lock):
        raise ValueError("V5 scoring was not bound to this protocol")
    if scoring.get("private_outcomes_read") is not False:
        raise ValueError("V5 scoring boundary was violated")
    if scoring.get("evaluation_role") != args.evaluation_role:
        raise ValueError("V5 scoring/evaluation role mismatch")
    if scoring.get("prediction_sha256", {}).get(args.method) != file_sha256(
        args.predictions
    ):
        raise ValueError("V5 prediction identity mismatch")
    if scoring.get("public_events_sha256") != file_sha256(args.public_events):
        raise ValueError("V5 public event identity mismatch")

    predictions = read_jsonl(args.predictions)
    public_rows = read_jsonl(args.public_events)
    private_rows = read_jsonl(args.private_outcomes)
    public_by_id = {str(row["event_id"]): row for row in public_rows}
    private_by_id = {str(row["event_id"]): row for row in private_rows}
    prediction_by_id = {str(row["event_id"]): row for row in predictions}
    exact_ids = set(public_by_id) == set(private_by_id) == set(prediction_by_id)
    if not exact_ids or len(prediction_by_id) != len(predictions):
        raise ValueError("V5 public/private/prediction event ids differ")

    results = []
    for event_id in sorted(prediction_by_id):
        prediction = prediction_by_id[event_id]
        public = public_by_id[event_id]
        private = private_by_id[event_id]
        selected = str(prediction["selected"])
        target = desired_candidate(str(private["decision"]))
        branch = private[f"branch_{selected}"]
        results.append(
            {
                "event_id": event_id,
                "task_id_hash": str(public["task_id_hash"]),
                "mode": public["mode"],
                "replay_valid": bool(private["replay_valid"]),
                "decision": str(private["decision"]),
                "decision_basis": private["decision_basis"],
                "mask_selected": prediction["mask_selected"],
                "selected": selected,
                "target": target,
                "causal_correct": selected == target,
                "flipped": bool(prediction["flipped"]),
                "flip_probability": prediction.get("flip_probability"),
                "margin_b_over_a": float(prediction["margin_b_over_a"]),
                "selected_terminal_similarity": branch["terminal_similarity"],
                "selected_progress_auc": branch["progress_auc"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "task_results.jsonl"
    write_jsonl(result_path, results)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_v5_offline_router_evaluation",
        "method": args.method,
        "evaluation_role": args.evaluation_role,
        "event_set_hash": event_set_hash(public_rows),
        "public_events_sha256": file_sha256(args.public_events),
        "private_outcomes_sha256": file_sha256(args.private_outcomes),
        "predictions_sha256": file_sha256(args.predictions),
        "scoring_summary_sha256": file_sha256(args.scoring_summary),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "results_sha256": file_sha256(result_path),
        **summarize_router_predictions(results),
        "public_only_model_scoring": True,
        "offline_outcomes_joined_after_scoring": True,
        "scope": (
            "known ToolSandbox development router diagnostic"
            if args.evaluation_role == "development"
            else "known ToolSandbox post-hoc router diagnostic"
        ),
    }
    write_json(args.output_dir / "eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
