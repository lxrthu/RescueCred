#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from rescuecredit.edit_credit import (
    canonical_action_edits,
    edit_comparison_prompt,
    edit_value_completion,
    fold_role,
    parse_public_preference_prompt,
    select_rescue_constrained_threshold,
    summarize_selection,
)
from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import canonical_completion
from scripts.freeze_toolsandbox_editcredit_protocol import STATUS
from scripts.train_route_a_preference import mean_completion_logprob


def _edit_order_margin(model, tokenizer, row, max_length: int, device, swapped: bool) -> float:
    payload = parse_public_preference_prompt(str(row["prompt"]))
    margins = []
    for edit in canonical_action_edits(row["action_a"], row["action_b"]):
        prompt = edit_comparison_prompt(
            public_payload=payload, edit=edit, swap_candidates=swapped
        )
        logp_a = mean_completion_logprob(
            model,
            tokenizer,
            prompt,
            edit_value_completion(edit.value_a),
            max_length,
            device,
        )
        logp_b = mean_completion_logprob(
            model,
            tokenizer,
            prompt,
            edit_value_completion(edit.value_b),
            max_length,
            device,
        )
        margins.append(float((logp_b - logp_a).detach()))
    return sum(margins) / len(margins)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("full_action", "editcredit"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--checkpoint-presentations", type=int, default=-1)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run_summary.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != STATUS:
        raise ValueError("invalid EditCredit protocol")
    if run.get("status") != "completed" or run.get("method") != args.method or run.get("fold") != args.fold:
        raise ValueError("run summary method/fold mismatch")
    if run.get("protocol_lock_sha256") != file_sha256(args.protocol_lock):
        raise ValueError("run is not bound to protocol")
    checkpoint_presentations = (
        int(run["presentations"])
        if args.checkpoint_presentations < 0
        else args.checkpoint_presentations
    )
    checkpoint_tag = f"p{checkpoint_presentations:06d}"
    checkpoint = run.get("checkpoints", {}).get(checkpoint_tag)
    if not isinstance(checkpoint, dict):
        raise ValueError("requested checkpoint is absent from run summary")
    if args.base_only != (checkpoint_presentations == 0):
        raise ValueError("--base-only is required exactly for presentation zero")
    expected_adapter = checkpoint.get("adapter")
    if args.base_only:
        if args.adapter is not None or expected_adapter is not None:
            raise ValueError("base checkpoint cannot bind an adapter")
        adapter_sha256 = None
    else:
        if args.adapter is None or expected_adapter != str(args.adapter):
            raise ValueError("checkpoint adapter path mismatch")
        adapter_sha256 = directory_sha256(args.adapter)
        if checkpoint.get("adapter_sha256") != adapter_sha256:
            raise ValueError("checkpoint adapter identity mismatch")
    if file_sha256(args.train_file) != protocol.get("train_sha256"):
        raise ValueError("pair bank identity mismatch")
    if directory_sha256(args.model) != protocol.get("base_model_sha256"):
        raise ValueError("base model identity mismatch")
    if run.get("max_length") != args.max_length or run.get("fp32") != args.fp32:
        raise ValueError("evaluation length/precision differs from training")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, torch_dtype=dtype
    ).cuda()
    model = base.eval() if args.base_only else PeftModel.from_pretrained(base, args.adapter).eval()
    device = next(model.parameters()).device

    rows = read_jsonl(args.train_file)
    assignment = {str(key): int(value) for key, value in protocol["task_fold_assignment"].items()}
    selected_rows = [
        {
            "event_id": str(row["event_id"]),
            "task_id_hash": str(row["task_id_hash"]),
            "prompt": str(row["prompt"]),
            "action_a": row["action_a"],
            "action_b": row["action_b"],
            "role": fold_role(
                row,
                assignment=assignment,
                test_fold=args.fold,
                folds=int(protocol["config"]["folds"]),
            ),
        }
        for row in rows
        if fold_role(
            row,
            assignment=assignment,
            test_fold=args.fold,
            folds=int(protocol["config"]["folds"]),
        )
        in {"calibration", "test"}
    ]
    started = time.time()
    scores: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in selected_rows:
            role = str(row["role"])
            if args.method == "full_action":
                logp_a = mean_completion_logprob(
                    model,
                    tokenizer,
                    str(row["prompt"]),
                    canonical_completion(row["action_a"]),
                    args.max_length,
                    device,
                )
                logp_b = mean_completion_logprob(
                    model,
                    tokenizer,
                    str(row["prompt"]),
                    canonical_completion(row["action_b"]),
                    args.max_length,
                    device,
                )
                margin_left = margin_right = float((logp_b - logp_a).detach())
            else:
                margin_left = _edit_order_margin(
                    model, tokenizer, row, args.max_length, device, False
                )
                margin_right = _edit_order_margin(
                    model, tokenizer, row, args.max_length, device, True
                )
            scores.append(
                {
                    "event_id": str(row["event_id"]),
                    "task_id_hash": str(row["task_id_hash"]),
                    "fold": args.fold,
                    "role": role,
                    "margin_b_over_a": 0.5 * (margin_left + margin_right),
                    "margin_original_order": margin_left,
                    "margin_swapped_order": margin_right,
                    "swap_consistent": (margin_left >= 0.0) == (margin_right >= 0.0),
                }
            )
    protected = {"decision", "replay_valid", "branch_a", "branch_b", "target", "correct"}
    if any(protected & set(row) for row in scores):
        raise RuntimeError("public score artifact contains protected outcome fields")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "scores.public.jsonl"
    write_jsonl(scores_path, scores)

    labels = {str(row["event_id"]): str(row["decision"]) for row in rows}
    predictions = [{**row, "decision": labels[str(row["event_id"])]} for row in scores]
    calibration = [row for row in predictions if row["role"] == "calibration"]
    test = [row for row in predictions if row["role"] == "test"]
    threshold = 0.0
    calibration_choice = None
    if args.method == "editcredit":
        calibration_choice = select_rescue_constrained_threshold(
            calibration, rescue_delta=float(protocol["config"]["rescue_delta"])
        )
        threshold = calibration_choice.threshold
    test_summary = summarize_selection(test, threshold=threshold)
    predictions_path = args.output_dir / "predictions.joined.jsonl"
    write_jsonl(predictions_path, predictions)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_editcredit_crossfit_evaluation",
        "method": args.method,
        "fold": args.fold,
        "checkpoint_presentations": checkpoint_presentations,
        "checkpoint_tag": checkpoint_tag,
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "run_summary_sha256": file_sha256(args.run_summary),
        "adapter_sha256": adapter_sha256,
        "public_scores_sha256": file_sha256(scores_path),
        "predictions_sha256": file_sha256(predictions_path),
        "calibration_events": len(calibration),
        "test_events": len(test),
        "selected_threshold": None if math.isinf(threshold) and threshold < 0 else threshold,
        "calibration_constraint": (
            {
                "rescue_drop": calibration_choice.rescue_drop,
                "reverse_recall": calibration_choice.reverse_recall,
                "route_to_a": calibration_choice.route_to_a,
                "feasible": calibration_choice.feasible,
            }
            if calibration_choice is not None
            else None
        ),
        "test": {key: value for key, value in test_summary.items() if key != "rows"},
        "swap_consistency": sum(row["swap_consistent"] for row in test) / max(1, len(test)),
        "ground_truth_source": "exact replay-valid frozen paired branch decision",
        "outcomes_joined_after_scoring": True,
        "claim_boundary": protocol["claim_boundary"],
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
