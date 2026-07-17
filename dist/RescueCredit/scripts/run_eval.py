#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from environments.api_bank import APIBankControlledEnv, APIBankHarness
from environments.api_bank.adapter import canonical_action
from rescuecredit.logging import JsonlLogger, write_json
from rescuecredit.training import build_prompt, parse_action


def load_model(checkpoint: str, revision: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, token=os.getenv("HF_TOKEN") or None)
    try:
        from peft import AutoPeftModelForCausalLM, PeftConfig

        PeftConfig.from_pretrained(checkpoint)
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="auto", token=os.getenv("HF_TOKEN") or None
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, revision=revision, torch_dtype=torch.bfloat16, device_map="auto", token=os.getenv("HF_TOKEN") or None
        )
    return model, tokenizer


def rollout(model, tokenizer, task: dict, condition: str, seed: int, max_new_tokens: int, logger: JsonlLogger) -> dict:
    import torch

    torch.manual_seed(seed)
    env = APIBankControlledEnv(task["max_steps"])
    env.reset(task, seed)
    harness = APIBankHarness(condition)
    history: list[dict] = []
    first_pass_valid_calls = 0
    total_reference_calls = len(task["reference_actions"])
    interventions = 0
    reward = 0.0
    while not env.done:
        prompt = build_prompt(task, history)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        proposal = parse_action(completion) or {"type": "invalid_output", "raw": completion}
        expected = env.expected_action()
        exact_first = expected is not None and canonical_action(proposal) == canonical_action(expected)
        first_pass_valid_calls += int(exact_first)
        executed, decision = harness.execute(env.observation(), proposal, expected)
        interventions += int(decision.triggered and decision.changes_execution)
        _, reward, _, info = env.step(executed)
        history.append(
            {
                "policy_proposal": proposal,
                "harness_patch": decision.patch_id if decision.triggered else None,
                "executed_action": executed,
                "tool_result": info.get("tool_result"),
            }
        )
        logger.write(
            {
                "task_id": task["task_id"],
                "condition": condition,
                "step": env.steps - 1,
                "proposal_action": proposal,
                "executed_action": executed,
                "patch_id": decision.patch_id,
                "ground_truth_match": exact_first,
                "terminal_reason": info["terminal_reason"],
            }
        )
    return {
        "success": bool(reward),
        "first_pass": first_pass_valid_calls / max(1, total_reference_calls),
        "intervened": interventions > 0,
        "interventions": interventions,
        "environment_steps": env.steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["dev", "test_id", "test_tool_ood"], default="test_id")
    parser.add_argument("--data-dir", type=Path, default=Path("data/api_bank_controlled_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in (args.data_dir / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        tasks = tasks[: args.limit]
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in (args.output_dir / "trajectory.jsonl", args.output_dir / "task_results.jsonl"):
        stale.unlink(missing_ok=True)
    step_logger = JsonlLogger(args.output_dir / "trajectory.jsonl")
    task_logger = JsonlLogger(args.output_dir / "task_results.jsonl")
    model, tokenizer = load_model(args.checkpoint, args.model_revision)
    model.eval()
    start = time.time()
    records = []
    for index, task in enumerate(tasks):
        common_seed = args.seed * 1_000_003 + index
        on = rollout(model, tokenizer, task, "H3", common_seed, args.max_new_tokens, step_logger)
        off = rollout(model, tokenizer, task, "H0", common_seed, args.max_new_tokens, step_logger)
        record = {
            "task_id": task["task_id"],
            "s_on": float(on["success"]),
            "s_off": float(off["success"]),
            "first_pass": off["first_pass"],
            "intervened": on["intervened"],
            "harness_on_steps": on["environment_steps"],
            "harness_off_steps": off["environment_steps"],
        }
        records.append(record)
        task_logger.write(record)
    count = max(1, len(records))
    s_on = sum(record["s_on"] for record in records) / count
    s_off = sum(record["s_off"] for record in records) / count
    config_payload = json.dumps(vars(args), default=str, sort_keys=True)
    summary = {
        "method": args.method,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "model_revision": args.model_revision,
        "split": args.split,
        "split_hash": manifest["split_hashes"][args.split],
        "config_hash": hashlib.sha256(config_payload.encode()).hexdigest(),
        "num_tasks": len(records),
        "s_on": s_on,
        "s_off": s_off,
        "dependence_gap": s_on - s_off,
        "first_pass": sum(record["first_pass"] for record in records) / count,
        "intervention_rate": sum(record["intervened"] for record in records) / count,
        "evaluation_steps": sum(record["harness_on_steps"] + record["harness_off_steps"] for record in records),
        "wall_time_sec": time.time() - start,
        "ground_truth_source": "frozen API-Bank-derived programmatic state checker",
    }
    write_json(args.output_dir / "eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
