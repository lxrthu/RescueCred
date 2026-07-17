#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from environments.api_bank import APIBankControlledEnv, APIBankHarness, APIBankVerifier
from environments.api_bank.adapter import canonical_action
from environments.api_bank.shadow import APIBankShadowRunner
from rescuecredit.accounting import BudgetCounter
from rescuecredit.audit import AuditLedger, UniformAuditScheduler
from rescuecredit.correction_preference import eligible_correction
from rescuecredit.engine import RescueCreditEngine
from rescuecredit.estimators import PatchEMA
from rescuecredit.logging import JsonlLogger, write_json
from rescuecredit.training import build_prompt, group_normalize, parse_action
from rescuecredit.types import RescueEvent, TokenSpan


@dataclass
class StepSample:
    input_ids: Any
    prompt_length: int
    old_logprobs: Any
    reference_logprobs: Any
    step_index: int


@dataclass
class CandidateTrace:
    steps: list[StepSample]
    assisted_return: float
    g0_hat: float
    event_step: int | None
    event: RescueEvent | None
    preference: tuple[str, str, str] | None


def read_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def completion_token_logprobs(model, input_ids, prompt_length: int):
    import torch

    config = model.config if hasattr(model, "config") else model.module.config
    attention_mask = input_ids.ne(config.pad_token_id)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs[:, prompt_length - 1 :]


def conditional_logprob(model, tokenizer, prompts: list[str], completions: list[str], device):
    import torch

    encoded_prompts = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=True)
    full = tokenizer([prompt + completion for prompt, completion in zip(prompts, completions)], padding=True, return_tensors="pt", add_special_tokens=True)
    full_ids = full.input_ids.to(device)
    attention = full.attention_mask.to(device)
    prompt_lengths = encoded_prompts.attention_mask.sum(-1).to(device)
    logits = model(input_ids=full_ids, attention_mask=attention).logits[:, :-1, :]
    labels = full_ids[:, 1:]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(labels.shape[1], device=device).unsqueeze(0)
    mask = positions.ge(prompt_lengths.unsqueeze(1) - 1) & attention[:, 1:].bool()
    return (token_logprobs * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def reference_context(unwrapped_model, reference_model):
    if reference_model is not None:
        return contextlib.nullcontext(reference_model)
    if hasattr(unwrapped_model, "disable_adapter"):
        return unwrapped_model.disable_adapter()
    raise RuntimeError("A frozen reference policy is required; use --use-lora or provide a base model")


def generate_step(model, reference_model, tokenizer, prompt: str, seed: int, args, accelerator, store_training: bool = True):
    import torch

    unwrapped = accelerator.unwrap_model(model)
    # GRPO's behavior-policy denominator must be deterministic.  Eval mode
    # disables LoRA dropout while still allowing gradients in the later loss.
    unwrapped.eval()
    if reference_model is not None:
        reference_model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
    prompt_length = inputs.input_ids.shape[1]
    torch.manual_seed(seed)
    with torch.no_grad():
        generated = unwrapped.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.95,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        if store_training:
            old_logprobs = completion_token_logprobs(unwrapped, generated, prompt_length).detach().cpu()
            with reference_context(unwrapped, reference_model) as reference:
                reference = reference if reference_model is not None else unwrapped
                reference_logprobs = completion_token_logprobs(reference, generated, prompt_length).detach().cpu()
        else:
            old_logprobs = reference_logprobs = None
    completion = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True)
    proposal = parse_action(completion) or {"type": "invalid_output", "raw": completion}
    sample = None
    if store_training:
        sample = StepSample(generated.detach().cpu(), prompt_length, old_logprobs, reference_logprobs, -1)
    return proposal, completion, sample


def rollout_candidate(model, reference_model, tokenizer, task: dict, update: int, candidate_index: int, args, accelerator, engine, budget, event_logger):
    env = APIBankControlledEnv(
        task["max_steps"], snapshot_dir=args.output_dir / "snapshots" / f"rank{accelerator.process_index}"
    )
    seed = args.seed * 10_000_019 + update * 10_007 + accelerator.process_index * 997 + candidate_index
    env.reset(task, seed)
    harness = APIBankHarness("H3")
    verifier = APIBankVerifier()
    history: list[dict[str, Any]] = []
    samples: list[StepSample] = []
    first_event: RescueEvent | None = None
    event_step: int | None = None
    event_rng_state = None
    history_before_event: list[dict[str, Any]] = []
    reward = 0.0

    while not env.done:
        step_index = len(samples)
        prompt = build_prompt(task, history)
        proposal, completion, sample = generate_step(
            model, reference_model, tokenizer, prompt, seed + step_index, args, accelerator, store_training=True
        )
        sample.step_index = step_index
        samples.append(sample)
        observation = env.observation()
        expected = env.expected_action()
        state_ref = env.snapshot()
        rng_state = env.get_rng_state()
        executed, decision = harness.execute(observation, proposal, expected)
        original_verification = verifier.verify(observation, proposal)
        corrected_verified = bool(
            decision.changes_execution
            and expected is not None
            and decision.corrected_action is not None
            and canonical_action(decision.corrected_action) == canonical_action(expected)
        )
        if first_event is None and decision.triggered and decision.teachable_patch and decision.changes_execution:
            token_count = max(1, sample.old_logprobs.shape[-1])
            first_event = RescueEvent(
                run_id=args.run_id,
                episode_id=f"u{update}_r{accelerator.process_index}",
                group_id=f"u{update}",
                candidate_id=f"r{accelerator.process_index}_c{candidate_index}",
                step_id=step_index,
                state_ref=state_ref,
                state_hash=observation["state_hash"],
                proposal_text=completion,
                proposal_action=proposal,
                executed_action=executed,
                correction_text=json.dumps(decision.corrected_action, ensure_ascii=False, sort_keys=True) if decision.corrected_action else None,
                event_type=decision.event_type,
                patch_id=decision.patch_id,
                patch_version="controlled-v1",
                verifier_label=original_verification.score,
                verifier_confidence=original_verification.confidence,
                verifier_reason=original_verification.reason,
                deterministic_outcome=decision.deterministic_outcome,
                shadow_safe=decision.patch_id == "wrong_tool_replace",
                teachable_patch=True,
                permanent_safety_patch=False,
                intervention_step=step_index,
                token_spans=[TokenSpan(0, token_count, "policy", "prefix")],
                metadata={
                    "correction_verified": corrected_verified,
                    "correction_verifier_confidence": 1.0 if corrected_verified else 0.0,
                    "rng_state_committed": True,
                    "snapshot_digest": state_ref.split(":", 1)[1],
                    "generation_seed": seed + step_index,
                    "audit_seed": seed + 900_001,
                    "source_sample_id": task.get("source_sample_id"),
                },
            )
            event_step = step_index
            event_rng_state = rng_state
            history_before_event = list(history)
        _, reward, _, info = env.step(executed)
        if first_event is not None and first_event.step_id == step_index:
            receipt_ok = bool(info.get("tool_result", {}).get("status") == "ok")
            first_event.metadata["correction_receipt_verified"] = receipt_ok
        budget.charge_main(1)
        history.append(
            {
                "policy_proposal": proposal,
                "harness_patch": decision.patch_id if decision.triggered else None,
                "executed_action": executed,
                "tool_result": info.get("tool_result"),
            }
        )

    if first_event is None:
        return CandidateTrace(samples, float(reward), float(reward), None, None, None)

    first_event.assisted_return = float(reward)

    def shadow_factory():
        shadow_history = list(history_before_event)
        pending_history: dict[str, Any] = {
            "policy_proposal": first_event.proposal_action,
            "harness_patch": None,
            "executed_action": first_event.proposal_action,
            "tool_result": None,
        }
        shadow_harness = APIBankHarness("H3")

        def continuation(obs: dict, shadow_step: int, _disabled_patch_id: str, previous_tool_result: dict | None) -> dict:
            nonlocal pending_history
            pending_history["tool_result"] = previous_tool_result
            shadow_history.append(pending_history)
            prompt = build_prompt(task, shadow_history)
            proposal, _, _ = generate_step(
                model,
                reference_model,
                tokenizer,
                prompt,
                seed + 100_000 + shadow_step,
                args,
                accelerator,
                store_training=False,
            )
            expected = env.expected_action()
            executed, decision = shadow_harness.execute(obs, proposal, expected)
            pending_history = {
                "policy_proposal": proposal,
                "harness_patch": decision.patch_id if decision.triggered else None,
                "executed_action": executed,
                "tool_result": None,
            }
            return executed

        return APIBankShadowRunner(env).run(
            state_ref=first_event.state_ref,
            original_action=first_event.proposal_action,
            disabled_patch_id=first_event.patch_id,
            rng_state=event_rng_state,
            max_steps=args.max_shadow_steps,
            expected_state_hash=first_event.state_hash,
            continuation=continuation,
        )

    if args.method == "naive_h_grpo":
        first_event.g0_hat = float(reward)
    elif args.method == "mask_correction":
        first_event.g0_hat = 0.0
    elif args.method == "full_shadow":
        shadow = shadow_factory()
        if shadow.replay_valid:
            budget.charge_shadow(shadow.steps)
            first_event.shadow_return = shadow.return_value
            first_event.g0_hat = shadow.return_value
        else:
            budget.charge_failed_replay(shadow.steps)
            first_event.g0_hat = 0.0
    else:
        outcome = engine.estimate(first_event, audit_seed=seed + 900_001, shadow_factory=shadow_factory)
        if not outcome.identifiable:
            first_event.g0_hat = 0.0
    first_event.rescue_gain_hat = float(reward) - float(first_event.g0_hat)
    event_logger.write(first_event)
    preference = None
    if args.method in {"mask_correction", "rescuecredit"} and eligible_correction(first_event):
        preference = (build_prompt(task, history_before_event), first_event.correction_text or "", first_event.proposal_text)
    return CandidateTrace(samples, float(reward), float(first_event.g0_hat), event_step, first_event, preference)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory-level provenance-aware GRPO for RescueCredit")
    parser.add_argument("--method", choices=["naive_h_grpo", "mask_correction", "rescuecredit", "full_shadow"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    parser.add_argument("--train-file", type=Path, default=Path("data/api_bank_controlled_v1/train.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/api_bank_controlled_v1/manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-shadow-steps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--policy-epochs", type=int, default=2)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--lambda-corr", type=float, default=0.1)
    parser.add_argument("--audit-probability", type=float, default=0.2)
    parser.add_argument("--total-interaction-budget", type=int, default=50000)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_id = args.run_id or f"{args.method}_seed{args.seed}"

    tasks = read_tasks(args.train_file)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not tasks:
        raise RuntimeError("training split has no tasks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = json.dumps(vars(args), default=str, sort_keys=True)
    config_hash = hashlib.sha256(config_payload.encode()).hexdigest()
    if args.dry_run:
        summary = {
            "status": "DRY_RUN_OK",
            "method": args.method,
            "tasks": len(tasks),
            "model": args.model,
            "split_hash": manifest["split_hashes"]["train"],
            "config_hash": config_hash,
        }
        write_json(args.output_dir / "dry_run.json", summary)
        print(json.dumps(summary, indent=2))
        return

    import torch
    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM, AutoTokenizer

    accelerator = Accelerator(mixed_precision="bf16")
    random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, token=os.getenv("HF_TOKEN") or None)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, torch_dtype=torch.bfloat16, token=os.getenv("HF_TOKEN") or None
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    reference_model = None
    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        base_model = get_peft_model(
            base_model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                # Zero is intentional: rollout old-logprobs and first-epoch
                # current logprobs must share an exact behavior policy.
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    else:
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.model_revision, torch_dtype=torch.bfloat16, token=os.getenv("HF_TOKEN") or None
        ).to(accelerator.device)
        reference_model.config.pad_token_id = tokenizer.pad_token_id
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((parameter for parameter in base_model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    model, optimizer = accelerator.prepare(base_model, optimizer)
    budget = BudgetCounter()
    audit_ledger = AuditLedger(args.output_dir / f"audit_rank{accelerator.process_index}.jsonl")
    engine = RescueCreditEngine(
        PatchEMA(beta=0.95),
        UniformAuditScheduler(args.audit_probability),
        audit_ledger,
        budget,
        exact_confidence_threshold=0.99,
    )
    event_logger = JsonlLogger(args.output_dir / f"rescue_events_rank{accelerator.process_index}.jsonl")
    train_logger = JsonlLogger(args.output_dir / f"train_rank{accelerator.process_index}.jsonl")
    global_counts = torch.zeros(3, dtype=torch.long, device=accelerator.device)
    start_time = time.time()

    for update in range(args.max_updates):
        max_cost_per_candidate = max(task["max_steps"] for task in tasks) * (2 if args.method in {"rescuecredit", "full_shadow"} else 1)
        reservation = max_cost_per_candidate * args.group_size * accelerator.num_processes
        if int(global_counts.sum().item()) + reservation > args.total_interaction_budget:
            break
        task = tasks[(update * accelerator.num_processes + accelerator.process_index) % len(tasks)]
        before = (budget.main_steps, budget.shadow_steps, budget.failed_replay_steps)
        traces = [
            rollout_candidate(
                model, reference_model, tokenizer, task, update, candidate, args, accelerator, engine, budget, event_logger
            )
            for candidate in range(args.group_size)
        ]
        local_delta = torch.tensor(
            [budget.main_steps - before[0], budget.shadow_steps - before[1], budget.failed_replay_steps - before[2]],
            dtype=torch.long,
            device=accelerator.device,
        )
        global_counts += accelerator.reduce(local_delta, reduction="sum")
        gh_advantages = group_normalize([trace.assisted_return for trace in traces])
        g0_advantages = group_normalize([trace.g0_hat for trace in traces])

        for policy_epoch in range(args.policy_epochs):
            model.eval()
            token_losses = []
            kl_terms = []
            for candidate_index, trace in enumerate(traces):
                for step in trace.steps:
                    ids = step.input_ids.to(accelerator.device)
                    current = completion_token_logprobs(model, ids, step.prompt_length)
                    old = step.old_logprobs.to(accelerator.device)
                    reference = step.reference_logprobs.to(accelerator.device)
                    if args.method == "naive_h_grpo" or trace.event_step is None:
                        advantage = gh_advantages[candidate_index]
                    elif args.method == "mask_correction" and step.step_index <= trace.event_step:
                        advantage = 0.0
                    elif step.step_index <= trace.event_step:
                        advantage = g0_advantages[candidate_index]
                    else:
                        advantage = gh_advantages[candidate_index]
                    ratio = torch.exp(current - old)
                    scalar = torch.tensor(advantage, dtype=current.dtype, device=current.device)
                    unclipped = ratio * scalar
                    clipped = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps) * scalar
                    token_losses.append(-torch.minimum(unclipped, clipped).mean())
                    log_ratio = reference - current
                    kl_terms.append((torch.exp(log_ratio) - log_ratio - 1.0).mean())
            loss_grpo = torch.stack(token_losses).mean()
            loss_kl = torch.stack(kl_terms).mean()
            preferences = [trace.preference for trace in traces if trace.preference is not None]
            loss_corr = torch.zeros((), device=accelerator.device)
            if preferences:
                prompts = [item[0] for item in preferences]
                chosen = [item[1] for item in preferences]
                rejected = [item[2] for item in preferences]
                chosen_logp = conditional_logprob(model, tokenizer, prompts, chosen, accelerator.device)
                rejected_logp = conditional_logprob(model, tokenizer, prompts, rejected, accelerator.device)
                loss_corr = -torch.nn.functional.logsigmoid(chosen_logp - rejected_logp).mean()
            loss = loss_grpo + args.kl_coef * loss_kl + args.lambda_corr * loss_corr
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_logger.write(
            {
                "update": update,
                "rank": accelerator.process_index,
                "method": args.method,
                "task_id": task["task_id"],
                "config_hash": config_hash,
                "global_main_steps": int(global_counts[0].item()),
                "global_shadow_steps": int(global_counts[1].item()),
                "global_failed_replay_steps": int(global_counts[2].item()),
                "global_training_steps": int(global_counts.sum().item()),
                "loss": float(loss.detach()),
                "loss_grpo": float(loss_grpo.detach()),
                "loss_kl": float(loss_kl.detach()),
                "loss_corr": float(loss_corr.detach()),
                "assisted_returns": [trace.assisted_return for trace in traces],
                "g0_hat": [trace.g0_hat for trace in traces],
            }
        )
        if accelerator.is_main_process and (update + 1) % args.save_every == 0:
            checkpoint = args.output_dir / "checkpoints" / f"update_{update + 1}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(checkpoint, save_function=accelerator.save)
            tokenizer.save_pretrained(checkpoint)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = args.output_dir / "checkpoints" / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(final_dir)
        write_json(
            args.output_dir / "run_summary.json",
            {
                "method": args.method,
                "seed": args.seed,
                "model": args.model,
                "model_revision": args.model_revision,
                "split_hash": manifest["split_hashes"]["train"],
                "config_hash": config_hash,
                "world_size": accelerator.num_processes,
                "main_steps": int(global_counts[0].item()),
                "shadow_steps": int(global_counts[1].item()),
                "failed_replay_steps": int(global_counts[2].item()),
                "total_training_steps": int(global_counts.sum().item()),
                "interaction_budget": args.total_interaction_budget,
                "budget_unused": args.total_interaction_budget - int(global_counts.sum().item()),
                "comparability": {
                    "group_size": args.group_size,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "max_shadow_steps": args.max_shadow_steps,
                    "interaction_budget": args.total_interaction_budget,
                    "learning_rate": args.learning_rate,
                    "clip_eps": args.clip_eps,
                    "policy_epochs": args.policy_epochs,
                    "kl_coef": args.kl_coef,
                    "lambda_corr": args.lambda_corr,
                    "audit_probability": args.audit_probability,
                    "use_lora": args.use_lora,
                },
                "wall_time_sec": time.time() - start_time,
                "checkpoint": str(final_dir),
                "status": "completed",
                "runtime": {
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                },
            },
        )


if __name__ == "__main__":
    main()
