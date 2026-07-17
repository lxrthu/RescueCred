#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from environments.api_bank import (
    APIBankControlledEnv,
    DeployableAPIBankHarness,
    FrozenModelCorrectionGenerator,
    OracleAPIBankHarness,
    merge_visible_tool_context,
    public_harness_observation,
)
from environments.api_bank.adapter import canonical_action
from environments.api_bank.data import digest_records
from rescuecredit.frozen_bank import directory_sha256, file_sha256
from rescuecredit.logging import JsonlLogger, write_json
from rescuecredit.training import build_prompt, history_patch_id, parse_action
from run_train import conditional_logprob


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


def rollout(
    model,
    tokenizer,
    task: dict,
    condition: str,
    seed: int,
    max_new_tokens: int,
    logger: JsonlLogger,
    harness_mode: str = "oracle",
    correction_generator=None,
    log_preference_margins: bool = False,
) -> dict:
    import torch

    torch.manual_seed(seed)
    env = APIBankControlledEnv(task["max_steps"])
    env.reset(task, seed)
    harness = (
        DeployableAPIBankHarness(condition, correction_generator=correction_generator)
        if harness_mode == "deployable"
        else OracleAPIBankHarness(condition)
    )
    history: list[dict] = []
    first_pass_valid_calls = 0
    total_reference_calls = len(task["reference_actions"])
    first_attempt_exact_calls = 0
    attempted_call_indices: set[int] = set()
    interventions = 0
    feedback_events = 0
    reward = 0.0
    previous_tool_result = None
    preference_margins: list[float] = []
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
        proposal_matches_expected = expected is not None and canonical_action(proposal) == canonical_action(expected)
        first_pass_valid_calls += int(proposal_matches_expected)
        call_index = env.call_index
        if expected is not None and call_index not in attempted_call_indices:
            attempted_call_indices.add(call_index)
            first_attempt_exact_calls += int(proposal_matches_expected)
        if harness_mode == "deployable":
            executed, decision = harness.execute(
                public_harness_observation(env.observation()), proposal, previous_tool_result
            )
        else:
            executed, decision = harness.execute(env.observation(), proposal, expected)
        executed_matches_expected = expected is not None and canonical_action(executed) == canonical_action(expected)
        feedback_events += int(decision.triggered)
        interventions += int(decision.triggered and decision.changes_execution)
        preference_margin = None
        if log_preference_margins and decision.triggered and decision.changes_execution:
            action_a = json.dumps(proposal, ensure_ascii=False, sort_keys=True)
            action_b = json.dumps(executed, ensure_ascii=False, sort_keys=True)
            with torch.no_grad():
                logp_a = conditional_logprob(
                    model, tokenizer, [prompt], [action_a], model.device
                )
                logp_b = conditional_logprob(
                    model, tokenizer, [prompt], [action_b], model.device
                )
            preference_margin = float((logp_b - logp_a).detach().mean())
            preference_margins.append(preference_margin)
        _, reward, _, info = env.step(executed)
        previous_tool_result = merge_visible_tool_context(
            previous_tool_result, info.get("tool_result")
        )
        history.append(
            {
                "policy_proposal": proposal,
                "harness_patch": history_patch_id(decision),
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
                # Keep the historical field name, but make its semantics match
                # the action that actually reached the environment.
                "ground_truth_match": executed_matches_expected,
                "proposal_ground_truth_match": proposal_matches_expected,
                "executed_ground_truth_match": executed_matches_expected,
                "patch_applied": decision.patch_id if decision.triggered and decision.changes_execution else None,
                "preference_margin_b_over_a": preference_margin,
                "terminal_reason": info["terminal_reason"],
            }
        )
    return {
        "success": bool(reward),
        "first_pass": first_pass_valid_calls / max(1, total_reference_calls),
        "first_attempt_accuracy": first_attempt_exact_calls
        / max(1, total_reference_calls),
        "intervened": interventions > 0,
        "interventions": interventions,
        "feedback_triggered": feedback_events > 0,
        "feedback_events": feedback_events,
        "environment_steps": env.steps,
        "preference_margins_b_over_a": preference_margins,
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
    parser.add_argument("--harness-mode", choices=["oracle", "deployable"], default="oracle")
    parser.add_argument("--harness-generator-model", default=None)
    parser.add_argument("--harness-generator-revision", default=None)
    parser.add_argument("--log-preference-margins", action="store_true")
    parser.add_argument("--experiment-protocol-lock", type=Path)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in (args.data_dir / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        tasks = tasks[: args.limit]
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    if digest_records(tasks) != manifest.get("split_hashes", {}).get(args.split):
        raise ValueError("evaluation split digest does not match the supplied manifest")
    protocol_lock_sha256 = None
    if args.experiment_protocol_lock is not None:
        protocol = json.loads(
            args.experiment_protocol_lock.read_text(encoding="utf-8")
        )
        if protocol.get("status") != "frozen_before_training":
            raise ValueError("evaluation protocol was not frozen before training")
        expected = protocol.get("config", {})
        checks = {
            "method": args.method == "base_qwen"
            or args.method in protocol.get("methods", []),
            "seed": args.seed == expected.get("seed"),
            "split": args.split == "dev",
            "max_new_tokens": args.max_new_tokens
            == expected.get("eval_max_new_tokens"),
            "harness_mode": args.harness_mode
            == expected.get("eval_harness_mode"),
            "generation": expected.get("eval_generation")
            == "greedy_do_sample_false",
            "manifest": file_sha256(args.data_dir / "manifest.json")
            == protocol.get("data", {}).get("manifest_sha256"),
            "dev_file": file_sha256(args.data_dir / "dev.jsonl")
            == protocol.get("data", {}).get("dev_sha256"),
        }
        for source, expected_hash in protocol.get("source_sha256", {}).items():
            checks[f"source:{source}"] = file_sha256(Path(source)) == expected_hash
        if not all(checks.values()):
            raise ValueError(f"evaluation violates frozen protocol: {checks}")
        protocol_lock_sha256 = file_sha256(args.experiment_protocol_lock)
    if args.harness_mode == "deployable" and manifest.get("available_tools_contract", {}).get(
        "all_runtime_tool_sets_reference_independent"
    ) is not True:
        raise ValueError(
            "deployable evaluation requires a manifest-certified reference-independent available_tools set"
        )
    if args.harness_mode == "deployable" and not all(
        task.get("available_tools_reference_independent") is True for task in tasks
    ):
        raise ValueError("deployable evaluation split contains uncertified available_tools")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in (args.output_dir / "trajectory.jsonl", args.output_dir / "task_results.jsonl"):
        stale.unlink(missing_ok=True)
    step_logger = JsonlLogger(args.output_dir / "trajectory.jsonl")
    task_logger = JsonlLogger(args.output_dir / "task_results.jsonl")
    model, tokenizer = load_model(args.checkpoint, args.model_revision)
    model.eval()
    correction_generator = None
    if args.harness_mode == "deployable":
        if not args.harness_generator_model:
            raise ValueError("deployable evaluation requires --harness-generator-model")
        correction_generator = FrozenModelCorrectionGenerator(
            args.harness_generator_model,
            revision=args.harness_generator_revision,
            device=str(model.device),
            max_new_tokens=64,
        )
    start = time.time()
    records = []
    for index, task in enumerate(tasks):
        common_seed = args.seed * 1_000_003 + index
        on = rollout(
            model,
            tokenizer,
            task,
            "H3",
            common_seed,
            args.max_new_tokens,
            step_logger,
            args.harness_mode,
            correction_generator,
            args.log_preference_margins,
        )
        off = rollout(
            model,
            tokenizer,
            task,
            "H0",
            common_seed,
            args.max_new_tokens,
            step_logger,
            args.harness_mode,
            correction_generator,
            False,
        )
        record = {
            "task_id": task["task_id"],
            "s_on": float(on["success"]),
            "s_off": float(off["success"]),
            "first_pass": off["first_pass"],
            "proposal_first_pass_on": on["first_pass"],
            "first_attempt_accuracy": off["first_attempt_accuracy"],
            "first_attempt_accuracy_on": on["first_attempt_accuracy"],
            "intervened": on["intervened"],
            "feedback_triggered": on["feedback_triggered"],
            "feedback_events": on["feedback_events"],
            "execution_interventions": on["interventions"],
            "harness_on_steps": on["environment_steps"],
            "harness_off_steps": off["environment_steps"],
            "preference_margins_b_over_a": on["preference_margins_b_over_a"],
        }
        records.append(record)
        task_logger.write(record)
    count = max(1, len(records))
    s_on = sum(record["s_on"] for record in records) / count
    s_off = sum(record["s_off"] for record in records) / count
    rescued_tasks = sum(
        record["s_on"] == 1.0 and record["s_off"] == 0.0 for record in records
    )
    harmed_tasks = sum(
        record["s_on"] == 0.0 and record["s_off"] == 1.0 for record in records
    )
    all_margins = [
        margin
        for record in records
        for margin in record["preference_margins_b_over_a"]
    ]
    config_payload = json.dumps(vars(args), default=str, sort_keys=True)
    summary = {
        "method": args.method,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "generation": "greedy_do_sample_false",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": directory_sha256(Path(args.checkpoint)),
        "experiment_protocol_lock_sha256": protocol_lock_sha256,
        "model_revision": args.model_revision,
        "split": args.split,
        "split_hash": manifest["split_hashes"][args.split],
        "manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "split_file_sha256": file_sha256(args.data_dir / f"{args.split}.jsonl"),
        "config_hash": hashlib.sha256(config_payload.encode()).hexdigest(),
        "num_tasks": len(records),
        "s_on": s_on,
        "s_off": s_off,
        "dependence_gap": s_on - s_off,
        "first_pass": sum(record["first_pass"] for record in records) / count,
        "proposal_first_pass_on": sum(
            record["proposal_first_pass_on"] for record in records
        ) / count,
        "first_attempt_accuracy": sum(
            record["first_attempt_accuracy"] for record in records
        ) / count,
        "first_attempt_accuracy_on": sum(
            record["first_attempt_accuracy_on"] for record in records
        ) / count,
        "first_pass_metric_note": (
            "first_attempt_accuracy is the strict metric; first_pass is the legacy "
            "completed-reference-call rate retained for backward compatibility"
        ),
        "intervention_rate": sum(record["intervened"] for record in records) / count,
        "feedback_task_rate": sum(record["feedback_triggered"] for record in records) / count,
        "feedback_events": sum(record["feedback_events"] for record in records),
        "execution_interventions": sum(record["execution_interventions"] for record in records),
        "rescued_tasks": rescued_tasks,
        "harmed_tasks": harmed_tasks,
        "both_success_tasks": sum(
            record["s_on"] == 1.0 and record["s_off"] == 1.0 for record in records
        ),
        "both_fail_tasks": sum(
            record["s_on"] == 0.0 and record["s_off"] == 0.0 for record in records
        ),
        "preference_margin_events": len(all_margins),
        "mean_preference_margin_b_over_a": (
            sum(all_margins) / len(all_margins) if all_margins else None
        ),
        "evaluation_steps": sum(record["harness_on_steps"] + record["harness_off_steps"] for record in records),
        "wall_time_sec": time.time() - start,
        "ground_truth_source": "frozen API-Bank-derived programmatic state checker",
        "harness_mode": args.harness_mode,
        "reference_free_intervention": args.harness_mode == "deployable",
        "evaluation_artifacts": [
            {
                "path": name,
                "sha256": file_sha256(args.output_dir / name),
                "rows": sum(
                    1
                    for line in (args.output_dir / name)
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ),
            }
            for name in ("trajectory.jsonl", "task_results.jsonl")
        ],
    }
    write_json(args.output_dir / "eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
