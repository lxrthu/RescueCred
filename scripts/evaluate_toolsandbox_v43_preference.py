#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import (
    canonical_completion,
    event_set_hash,
    summarize_evaluation_rows,
)
from scripts.train_route_a_preference import mean_completion_logprob
from scripts.train_toolsandbox_v43_preference import PROTOCOL_STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("mask", "v43"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--public-events", type=Path, required=True)
    parser.add_argument("--private-outcomes", type=Path, required=True)
    parser.add_argument("--evaluation-role", choices=("development", "confirmation"), required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run_summary.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("invalid frozen V4.3 protocol")
    if run.get("method") != args.method or run.get("status") != "completed":
        raise ValueError("run summary method/status mismatch")
    if run.get("protocol_lock_sha256") != file_sha256(args.protocol_lock):
        raise ValueError("adapter training was not bound to this protocol")
    if run.get("adapter") != str(args.adapter):
        raise ValueError("adapter path does not match its run summary")
    if run.get("adapter_sha256") != directory_sha256(args.adapter):
        raise ValueError("adapter content does not match its run summary")
    if run.get("max_length") != args.max_length or run.get("fp32") != args.fp32:
        raise ValueError("evaluation precision/length differs from training")
    if run.get("base_model_sha256") != directory_sha256(args.model):
        raise ValueError("base model differs from training")

    public_rows = read_jsonl(args.public_events)
    private_by_id = {
        str(row["event_id"]): row for row in read_jsonl(args.private_outcomes)
    }
    if len(private_by_id) != len(public_rows):
        raise ValueError("public/private evaluation row counts differ")
    if set(private_by_id) != {str(row["event_id"]) for row in public_rows}:
        raise ValueError("public/private evaluation event ids differ")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).cuda()
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    device = next(model.parameters()).device
    started = time.time()
    results = []
    with torch.no_grad():
        for public in public_rows:
            event_id = str(public["event_id"])
            private = private_by_id[event_id]
            logp_a = mean_completion_logprob(
                model,
                tokenizer,
                str(public["prompt"]),
                canonical_completion(public["action_a"]),
                args.max_length,
                device,
            )
            logp_b = mean_completion_logprob(
                model,
                tokenizer,
                str(public["prompt"]),
                canonical_completion(public["action_b"]),
                args.max_length,
                device,
            )
            margin_b = float((logp_b - logp_a).detach())
            selected = "b" if margin_b > 0.0 else "a"
            target = "b" if private["decision"] == "rescue_preference" else "a"
            branch = private[f"branch_{selected}"]
            results.append(
                {
                    "event_id": event_id,
                    "task_id_hash": public["task_id_hash"],
                    "mode": public["mode"],
                    "replay_valid": private["replay_valid"],
                    "decision": private["decision"],
                    "decision_basis": private["decision_basis"],
                    "selected": selected,
                    "target": target,
                    "causal_correct": selected == target,
                    "margin_b_over_a": margin_b,
                    "selected_terminal_similarity": branch["terminal_similarity"],
                    "selected_progress_auc": branch["progress_auc"],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "task_results.jsonl"
    write_jsonl(result_path, results)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_v43_preference_evaluation",
        "evaluation_role": args.evaluation_role,
        "method": args.method,
        "model": str(args.model),
        "base_model_sha256": directory_sha256(args.model),
        "adapter": str(args.adapter),
        "adapter_sha256": directory_sha256(args.adapter),
        "run_summary_sha256": file_sha256(args.run_summary),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "public_events_sha256": file_sha256(args.public_events),
        "private_outcomes_sha256": file_sha256(args.private_outcomes),
        "event_set_hash": event_set_hash(public_rows),
        "results_sha256": file_sha256(result_path),
        **summarize_evaluation_rows(results),
        "worker_receives_public_prompt_and_candidates_only": True,
        "offline_outcomes_joined_after_scoring": True,
        "scope": (
            "controlled-state ToolSandbox development preference diagnostic"
            if args.evaluation_role == "development"
            else "fresh controlled-state ToolSandbox confirmation diagnostic"
        ),
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
