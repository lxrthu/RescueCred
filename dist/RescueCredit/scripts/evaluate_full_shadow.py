#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from statistics import mean, pvariance

from environments.api_bank import APIBankControlledEnv, APIBankHarness
from environments.api_bank.shadow import APIBankShadowRunner
from rescuecredit.estimators import PatchEMA, residual_estimate
from rescuecredit.evaluation import mse, spearman
from rescuecredit.logging import JsonlLogger, write_json
from rescuecredit.training import build_prompt, parse_action


def load_model(checkpoint: str, revision: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, token=os.getenv("HF_TOKEN") or None)
    try:
        from peft import AutoPeftModelForCausalLM, PeftConfig

        PeftConfig.from_pretrained(checkpoint)
        model = AutoPeftModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, revision=revision, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, task: dict, history: list[dict], max_new_tokens: int) -> tuple[dict, str]:
    import torch

    prompt = build_prompt(task, history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return parse_action(text) or {"type": "invalid_output", "raw": text}, text


def pair_accuracy(estimates: list[float], truths: list[float]) -> float:
    pairs = [(i, j) for i in range(len(truths)) for j in range(i + 1, len(truths)) if truths[i] != truths[j]]
    if not pairs:
        return float("nan")
    return mean(float((estimates[i] - estimates[j]) * (truths[i] - truths[j]) > 0) for i, j in pairs)


def collect_event(
    model, tokenizer, task: dict, seed: int, max_new_tokens: int, max_shadow_steps: int, snapshot_dir: Path
):
    env = APIBankControlledEnv(task["max_steps"], snapshot_dir=snapshot_dir)
    env.reset(task, seed)
    harness = APIBankHarness("H3")
    history: list[dict] = []
    event = None
    main_steps = 0
    reward = 0.0
    while not env.done:
        proposal, text = generate(model, tokenizer, task, history, max_new_tokens)
        obs = env.observation()
        expected = env.expected_action()
        state_ref, rng_state = env.snapshot(), env.get_rng_state()
        executed, decision = harness.execute(obs, proposal, expected)
        if event is None and decision.triggered and decision.changes_execution and decision.teachable_patch:
            event = {
                "state_ref": state_ref,
                "state_hash": obs["state_hash"],
                "rng_state": rng_state,
                "proposal": proposal,
                "proposal_text": text,
                "patch_id": decision.patch_id,
                "history": list(history),
            }
        _, reward, _, info = env.step(executed)
        main_steps += 1
        history.append({"policy_proposal": proposal, "executed_action": executed, "tool_result": info.get("tool_result")})
    if event is None:
        return None

    shadow_history = list(event["history"])
    pending = {
        "policy_proposal": event["proposal"],
        "executed_action": event["proposal"],
        "tool_result": None,
    }
    shadow_harness = APIBankHarness("H3")

    def continuation(obs: dict, step: int, _disabled: str, previous_tool_result: dict | None) -> dict:
        nonlocal pending
        pending["tool_result"] = previous_tool_result
        shadow_history.append(pending)
        proposal, _ = generate(model, tokenizer, task, shadow_history, max_new_tokens)
        expected = env.expected_action()
        executed, decision = shadow_harness.execute(obs, proposal, expected)
        pending = {"policy_proposal": proposal, "executed_action": executed, "tool_result": None}
        return executed

    shadow = APIBankShadowRunner(env).run(
        event["state_ref"],
        event["proposal"],
        event["patch_id"],
        event["rng_state"],
        max_shadow_steps,
        expected_state_hash=event["state_hash"],
        continuation=continuation,
    )
    return {
        "task_id": task["task_id"],
        "patch_id": event["patch_id"],
        "gh": float(reward),
        "g0_truth": shadow.return_value if shadow.replay_valid else None,
        "main_steps": main_steps,
        "shadow_steps": shadow.steps,
        "replay_valid": shadow.replay_valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--data-file", type=Path, default=Path("data/api_bank_controlled_v1/full_shadow_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-shadow-steps", type=int, default=12)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in args.data_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(args.output_dir / "full_shadow_events.jsonl")
    model, tokenizer = load_model(args.checkpoint, args.model_revision)
    all_events = []
    events = []
    for index, task in enumerate(tasks):
        record = collect_event(
            model,
            tokenizer,
            task,
            args.seed + index,
            args.max_new_tokens,
            args.max_shadow_steps,
            args.output_dir / "snapshots",
        )
        if record is not None:
            all_events.append(record)
            logger.write(record)
            if record["replay_valid"] and record["g0_truth"] is not None:
                events.append(record)

    truths = [record["g0_truth"] for record in events]
    g0_variance = pvariance(truths) if len(truths) >= 2 else 0.0
    if g0_variance <= 1e-12:
        summary = {
            "checkpoint": args.checkpoint,
            "frozen_tasks": len(tasks),
            "intervention_events": len(events),
            "replay_failure_rate": sum(not record["replay_valid"] for record in all_events) / max(1, len(all_events)),
            "g0_variance": g0_variance,
            "identifiability_gate_pass": False,
            "failure": "G0 variance is zero; estimator comparison is not identifiable",
        }
        write_json(args.output_dir / "estimator_summary.json", summary)
        raise SystemExit("Full Shadow identifiability gate failed: Var(G0) == 0")
    results = []
    for probability in (0.05, 0.10, 0.20, 0.40, 1.0):
        ema = PatchEMA(beta=0.95)
        rng = random.Random(args.seed + int(probability * 1000))
        estimates = []
        audited_shadow_steps = 0
        for record in events:
            mu = ema.predict(record["patch_id"])
            draw = int(rng.random() < probability)
            estimate = residual_estimate(mu, draw, probability, record["g0_truth"] if draw else None)
            estimates.append(estimate)
            if draw:
                audited_shadow_steps += record["shadow_steps"]
                ema.update(record["patch_id"], record["g0_truth"])
        results.append(
            {
                "p": probability,
                "events": len(events),
                "bias": mean(estimate - truth for estimate, truth in zip(estimates, truths)) if events else None,
                "mae": mean(abs(estimate - truth) for estimate, truth in zip(estimates, truths)) if events else None,
                "mse": mse(estimates, truths),
                "rank_correlation": spearman(estimates, truths),
                "pair_accuracy": pair_accuracy(estimates, truths),
                "extra_step_ratio": audited_shadow_steps / max(1, sum(record["main_steps"] for record in events)),
            }
        )
    summary = {
        "checkpoint": args.checkpoint,
        "frozen_tasks": len(tasks),
        "intervention_events": len(events),
        "replay_failure_rate": sum(not record["replay_valid"] for record in all_events) / max(1, len(all_events)),
        "g0_variance": g0_variance,
        "identifiability_gate_pass": True,
        "estimators": results,
    }
    # Replay failures are also available from the event JSONL; avoid inventing missing events.
    summary["replay_valid_events"] = len(events)
    write_json(args.output_dir / "estimator_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
