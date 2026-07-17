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

from environments.api_bank import (
    APIBankControlledEnv,
    APIBankVerifier,
    DeployableAPIBankHarness,
    FrozenModelCorrectionGenerator,
    OracleAPIBankHarness,
    merge_visible_tool_context,
    public_harness_observation,
)
from environments.api_bank.adapter import canonical_action
from environments.api_bank.data import digest_records
from environments.api_bank.shadow import APIBankShadowRunner
from rescuecredit.accounting import BudgetCounter
from rescuecredit.audit import AuditLedger, UniformAuditScheduler
from rescuecredit.correction_preference import eligible_correction
from rescuecredit.engine import RescueCreditEngine
from rescuecredit.estimators import PatchEMA
from rescuecredit.frozen_bank import directory_sha256, file_sha256
from rescuecredit.logging import JsonlLogger, write_json
from rescuecredit.training import build_prompt, group_normalize, history_patch_id, parse_action
from rescuecredit.types import RescueEvent, TokenSpan
from rescuecredit.v2_preference import V2PreferenceDecision, decide_v2_preference
from rescuecredit.visible_curriculum import VisibleStructureCurriculum

DEPLOYABLE_METHODS = frozenset({"mask_correction_v2", "rescuecredit_v2"})


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
    v2_preference: V2PreferenceSample | None
    diagnostic_shadow_return: float | None
    diagnostic_shadow_steps: int
    diagnostic_replay_valid: bool | None
    truncated_for_budget: bool = False
    diagnostic_shadow_already_charged: bool = False


@dataclass
class V2PreferenceSample:
    prompt: str
    action_a: str
    action_b: str
    decision: V2PreferenceDecision
    event: RescueEvent


def read_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def completion_token_logprobs(model, input_ids, prompt_length: int):
    import torch

    config = model.config if hasattr(model, "config") else model.module.config
    attention_mask = input_ids.ne(config.pad_token_id)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()
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
    logits = model(input_ids=full_ids, attention_mask=attention).logits[:, :-1, :].float()
    labels = full_ids[:, 1:]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(labels.shape[1], device=device).unsqueeze(0)
    mask = positions.ge(prompt_lengths.unsqueeze(1) - 1) & attention[:, 1:].bool()
    return (token_logprobs * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def routed_advantage(
    method: str,
    event_step: int | None,
    step_index: int,
    assisted_advantage: float,
    shadow_advantage: float,
) -> float:
    if method == "naive_h_grpo" or event_step is None:
        return assisted_advantage
    if method in {"mask_correction", "mask_correction_v2", "rescuecredit_v2"} and step_index <= event_step:
        return 0.0
    if step_index <= event_step:
        return shadow_advantage
    return assisted_advantage


def should_stop_for_budget(
    budget_mode: str,
    global_main_steps: int,
    global_total_steps: int,
    main_target: int,
    total_target: int,
    next_total_reservation: int,
) -> bool:
    if budget_mode == "main":
        return global_main_steps >= main_target
    return global_total_steps + next_total_reservation > total_target


def execute_harness_action(
    method: str,
    env,
    observation: dict[str, Any],
    proposal: dict[str, Any],
    previous_tool_result: dict[str, Any] | None,
    oracle_harness: OracleAPIBankHarness,
    deployable_harness: DeployableAPIBankHarness | None,
):
    if method in DEPLOYABLE_METHODS:
        if deployable_harness is None:
            raise RuntimeError(f"{method} requires a deployable harness")
        public_observation = public_harness_observation(observation)
        executed, decision = deployable_harness.execute(
            public_observation, proposal, previous_tool_result
        )
        return executed, decision, None, public_observation
    expected = env.expected_action()
    executed, decision = oracle_harness.execute(observation, proposal, expected)
    return executed, decision, expected, observation


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


def rollout_candidate(
    model,
    reference_model,
    tokenizer,
    task: dict,
    update: int,
    candidate_index: int,
    args,
    accelerator,
    engine,
    budget,
    event_logger,
    deployable_harness: DeployableAPIBankHarness | None = None,
    main_step_limit: int | None = None,
):
    env = APIBankControlledEnv(
        task["max_steps"], snapshot_dir=args.output_dir / "snapshots" / f"rank{accelerator.process_index}"
    )
    seed = args.seed * 10_000_019 + update * 10_007 + accelerator.process_index * 997 + candidate_index
    env.reset(task, seed)
    oracle_harness = OracleAPIBankHarness("H3")
    verifier = APIBankVerifier()
    history: list[dict[str, Any]] = []
    samples: list[StepSample] = []
    first_event: RescueEvent | None = None
    event_step: int | None = None
    event_rng_state = None
    history_before_event: list[dict[str, Any]] = []
    reward = 0.0
    previous_tool_result: dict[str, Any] | None = None
    truncated_for_budget = False

    while not env.done:
        if main_step_limit is not None and budget.main_steps >= main_step_limit:
            truncated_for_budget = True
            break
        step_index = len(samples)
        prompt = build_prompt(task, history)
        proposal, completion, sample = generate_step(
            model, reference_model, tokenizer, prompt, seed + step_index, args, accelerator, store_training=True
        )
        sample.step_index = step_index
        samples.append(sample)
        observation = env.observation()
        state_ref = env.snapshot()
        rng_state = env.get_rng_state()
        a_validity = b_validity = None
        executed, decision, expected, verifier_observation = execute_harness_action(
            args.method,
            env,
            observation,
            proposal,
            previous_tool_result,
            oracle_harness,
            deployable_harness,
        )
        if args.method in DEPLOYABLE_METHODS:
            original_verification = verifier.verify(verifier_observation, proposal)
            if decision.corrected_action is not None:
                a_validity, b_validity = deployable_harness.validity_pair(
                    verifier_observation,
                    proposal,
                    decision.corrected_action,
                    previous_tool_result,
                )
            corrected_verified = bool(b_validity and b_validity.semantic_valid == "true")
        else:
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
                patch_version=(
                    "deployable-v2"
                    if args.method in DEPLOYABLE_METHODS
                    else "controlled-v1"
                ),
                verifier_label=original_verification.score,
                verifier_confidence=original_verification.confidence,
                verifier_reason=original_verification.reason,
                deterministic_outcome=decision.deterministic_outcome,
                shadow_safe=decision.patch_id
                in {
                    "wrong_tool_replace",
                    "semantic_argument_mismatch",
                    "visible_schema_repair",
                    "visible_argument_repair",
                    "visible_prerequisite_repair",
                    "generated_visible_schema_repair",
                    "generated_visible_argument_repair",
                },
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
                    "a_executable_valid": a_validity.executable_valid if a_validity else None,
                    "a_semantic_valid": a_validity.semantic_valid if a_validity else "unknown",
                    "b_executable_valid": b_validity.executable_valid if b_validity else None,
                    "b_semantic_valid": b_validity.semantic_valid if b_validity else "unknown",
                    "validator_source": (
                        "visible_context_rules_v1"
                        if args.method in DEPLOYABLE_METHODS
                        else "oracle_teacher"
                    ),
                    "reference_free_intervention": args.method
                    in DEPLOYABLE_METHODS,
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

    if truncated_for_budget:
        # The final environment step is counted so every arm reaches the exact
        # main-step budget, but an incomplete trajectory must never train the
        # policy or contribute a return label.
        return CandidateTrace(
            [], 0.0, 0.0, None, None, None, None, None, 0, None, True, False
        )

    if first_event is None:
        return CandidateTrace(
            samples, float(reward), float(reward), None, None, None, None,
            None, 0, None, False, False,
        )

    first_event.assisted_return = float(reward)

    shadow_cache = None

    def shadow_factory():
        nonlocal shadow_cache
        if shadow_cache is not None:
            return shadow_cache
        shadow_history = list(history_before_event)
        pending_history: dict[str, Any] = {
            "policy_proposal": first_event.proposal_action,
            "harness_patch": None,
            "executed_action": first_event.proposal_action,
            "tool_result": None,
        }
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
            pending_history = {
                "policy_proposal": proposal,
                "harness_patch": None,
                "executed_action": proposal,
                "tool_result": None,
            }
            # G0 is the fully unassisted potential return from the committed
            # pre-intervention state. Re-enabling the Harness here would map
            # both potential outcomes back to success and erase causal credit.
            return proposal

        shadow_cache = APIBankShadowRunner(env).run(
            state_ref=first_event.state_ref,
            original_action=first_event.proposal_action,
            disabled_patch_id=first_event.patch_id,
            rng_state=event_rng_state,
            max_steps=args.max_shadow_steps,
            expected_state_hash=first_event.state_hash,
            continuation=continuation,
        )
        return shadow_cache

    if args.method == "naive_h_grpo":
        first_event.g0_hat = float(reward)
    elif args.method in {"mask_correction", "mask_correction_v2"}:
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

    diagnostic_shadow_return = None
    diagnostic_shadow_steps = 0
    diagnostic_replay_valid = None
    diagnostic_shadow_already_charged = False
    if args.diagnostic_full_shadow:
        diagnostic_shadow = shadow_factory()
        diagnostic_shadow_steps = int(diagnostic_shadow.steps)
        diagnostic_replay_valid = bool(diagnostic_shadow.replay_valid)
        if diagnostic_shadow.replay_valid:
            diagnostic_shadow_return = float(diagnostic_shadow.return_value)
        diagnostic_shadow_already_charged = bool(
            args.method in {"rescuecredit", "full_shadow"}
            and first_event.audit_draw == 1
        )
        first_event.metadata.update(
            {
                "diagnostic_full_shadow": True,
                "diagnostic_shadow_return": diagnostic_shadow_return,
                "diagnostic_shadow_steps": diagnostic_shadow_steps,
                "diagnostic_replay_valid": diagnostic_replay_valid,
                "diagnostic_only_not_used_for_training": args.method
                in {"naive_h_grpo", "mask_correction"},
            }
        )
    first_event.rescue_gain_hat = float(reward) - float(first_event.g0_hat)
    preference = None
    v2_preference = None
    if args.method in {"mask_correction", "rescuecredit"} and eligible_correction(first_event):
        preference = (build_prompt(task, history_before_event), first_event.correction_text or "", first_event.proposal_text)
    elif args.method in DEPLOYABLE_METHODS and first_event.correction_text:
        decision = decide_v2_preference(first_event, max_causal_weight=args.max_causal_weight)
        first_event.metadata.update(
            {
                "delta": decision.delta,
                "causal_decision": decision.causal_decision,
                "causal_weight": decision.causal_weight,
                "ordinary_direction": decision.ordinary_direction,
                "causal_direction": decision.causal_direction,
            }
        )
        if decision.ordinary_direction is not None or decision.causal_direction is not None:
            v2_preference = V2PreferenceSample(
                prompt=build_prompt(task, history_before_event),
                action_a=json.dumps(first_event.proposal_action, ensure_ascii=False, sort_keys=True),
                action_b=json.dumps(first_event.executed_action, ensure_ascii=False, sort_keys=True),
                decision=decision,
                event=first_event,
            )
    event_logger.write(first_event)
    return CandidateTrace(
        samples,
        float(reward),
        float(first_event.g0_hat),
        event_step,
        first_event,
        preference,
        v2_preference,
        diagnostic_shadow_return,
        diagnostic_shadow_steps,
        diagnostic_replay_valid,
        False,
        diagnostic_shadow_already_charged,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory-level provenance-aware GRPO for RescueCredit")
    parser.add_argument(
        "--method",
        choices=[
            "naive_h_grpo",
            "mask_correction",
            "mask_correction_v2",
            "rescuecredit",
            "rescuecredit_v2",
            "full_shadow",
        ],
        required=True,
    )
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
    parser.add_argument("--lambda-causal", type=float, default=0.1)
    parser.add_argument("--preference-beta", type=float, default=1.0)
    parser.add_argument("--max-causal-weight", type=float, default=2.5)
    parser.add_argument("--audit-probability", type=float, default=0.2)
    parser.add_argument("--audit-warm-start-events", type=int, default=2)
    parser.add_argument("--total-interaction-budget", type=int, default=50000)
    parser.add_argument("--budget-mode", choices=["total", "main"], default="total")
    parser.add_argument("--main-interaction-budget", type=int, default=0)
    parser.add_argument("--harness-generator-model", default=None)
    parser.add_argument("--visible-curriculum-fraction", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--diagnostic-full-shadow",
        action="store_true",
        help="Replay every teachable intervention for diagnostics only; never changes Naive/Mask training credit.",
    )
    parser.add_argument(
        "--force-shadow-credit",
        action="store_true",
        help="Disable deterministic verifier shortcuts so RescueCredit trains on replayed G0.",
    )
    parser.add_argument(
        "--strict-main-budget",
        action="store_true",
        help="Stop at exactly --main-interaction-budget and exclude the final truncated rollout from learning.",
    )
    parser.add_argument(
        "--experiment-protocol-lock",
        type=Path,
        help="Frozen protocol JSON whose data/model/source identities bind this run.",
    )
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_id = args.run_id or f"{args.method}_seed{args.seed}"
    if args.budget_mode == "main" and args.main_interaction_budget <= 0:
        raise ValueError("--budget-mode main requires a positive --main-interaction-budget")
    if args.lambda_causal < 0 or args.preference_beta <= 0 or args.max_causal_weight <= 0:
        raise ValueError("causal loss coefficients must be non-negative and beta/weight must be positive")
    if args.force_shadow_credit and args.method not in {"rescuecredit", "full_shadow"}:
        raise ValueError("--force-shadow-credit is only valid for RescueCredit/Full-Shadow")
    if args.strict_main_budget and args.budget_mode != "main":
        raise ValueError("--strict-main-budget requires --budget-mode main")

    tasks = read_tasks(args.train_file)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not tasks:
        raise RuntimeError("training split has no tasks")
    if digest_records(tasks) != manifest.get("split_hashes", {}).get("train"):
        raise ValueError("train file digest does not match the supplied manifest")
    protocol_lock_sha256 = None
    if args.experiment_protocol_lock is not None:
        protocol = json.loads(
            args.experiment_protocol_lock.read_text(encoding="utf-8")
        )
        if protocol.get("status") != "frozen_before_training":
            raise ValueError("experiment protocol was not frozen before training")
        if args.method not in protocol.get("methods", []):
            raise ValueError(f"method {args.method!r} is not authorized by protocol")
        frozen_config = protocol.get("config", {})
        actual_config = {
            "seed": args.seed,
            "model_revision": args.model_revision,
            "max_updates": args.max_updates,
            "budget_mode": args.budget_mode,
            "main_interaction_budget": args.main_interaction_budget,
            "total_interaction_budget": args.total_interaction_budget,
            "group_size": args.group_size,
            "max_new_tokens": args.max_new_tokens,
            "max_shadow_steps": args.max_shadow_steps,
            "temperature": args.temperature,
            "policy_epochs": args.policy_epochs,
            "learning_rate": args.learning_rate,
            "clip_eps": args.clip_eps,
            "kl_coef": args.kl_coef,
            "audit_probability": args.audit_probability,
            "audit_warm_start_events": args.audit_warm_start_events,
            "lambda_corr": args.lambda_corr,
            "lambda_causal": args.lambda_causal,
            "visible_curriculum_fraction": args.visible_curriculum_fraction,
            "use_lora": args.use_lora,
            "fp32": args.fp32,
            "diagnostic_full_shadow": args.diagnostic_full_shadow,
            "strict_main_budget": args.strict_main_budget,
            "save_every": args.save_every,
        }
        mismatches = {
            key: {"actual": value, "expected": frozen_config.get(key)}
            for key, value in actual_config.items()
            if value != frozen_config.get(key)
        }
        if args.force_shadow_credit != (
            args.method == "rescuecredit"
            and frozen_config.get("force_shadow_credit_for_rescue") is True
        ):
            mismatches["force_shadow_credit"] = {
                "actual": args.force_shadow_credit,
                "expected": args.method == "rescuecredit",
            }
        if mismatches:
            raise ValueError(f"training config violates frozen protocol: {mismatches}")
        if protocol.get("data", {}).get("train_split_hash") != manifest.get(
            "split_hashes", {}
        ).get("train"):
            raise ValueError("protocol/train split identity mismatch")
        protocol_data = protocol.get("data", {})
        dev_file = args.train_file.parent / "dev.jsonl"
        if file_sha256(args.manifest) != protocol_data.get("manifest_sha256"):
            raise ValueError("protocol/manifest file hash mismatch")
        if file_sha256(args.train_file) != protocol_data.get("train_sha256"):
            raise ValueError("protocol/train file hash mismatch")
        if not dev_file.is_file() or file_sha256(dev_file) != protocol_data.get(
            "dev_sha256"
        ):
            raise ValueError("protocol/dev file hash mismatch")
        if directory_sha256(Path(args.model)) != protocol.get("base_model", {}).get(
            "directory_sha256"
        ):
            raise ValueError("protocol/base model identity mismatch")
        for source, expected_hash in protocol.get("source_sha256", {}).items():
            if file_sha256(Path(source)) != expected_hash:
                raise ValueError(f"source changed after protocol freeze: {source}")
        protocol_lock_sha256 = file_sha256(args.experiment_protocol_lock)
    if args.method in DEPLOYABLE_METHODS and manifest.get("available_tools_contract", {}).get(
        "all_runtime_tool_sets_reference_independent"
    ) is not True:
        raise ValueError(
            "deployable training requires a manifest-certified reference-independent available_tools set"
        )
    if args.method in DEPLOYABLE_METHODS and not all(
        task.get("available_tools_reference_independent") is True for task in tasks
    ):
        raise ValueError("deployable training task file contains uncertified available_tools")
    curriculum = VisibleStructureCurriculum(
        tasks,
        fraction=args.visible_curriculum_fraction,
        seed=args.seed,
    )
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
            "visible_curriculum_fraction": args.visible_curriculum_fraction,
            "visible_pool_size": curriculum.visible_pool_size,
            "visible_pool_hash": curriculum.visible_pool_hash,
            "visible_reason_pool_sizes": curriculum.reason_pool_sizes,
            "reference_free_curriculum": curriculum.reference_free_selection,
        }
        write_json(args.output_dir / "dry_run.json", summary)
        print(json.dumps(summary, indent=2))
        return

    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
    from datetime import timedelta
    from transformers import AutoModelForCausalLM, AutoTokenizer

    accelerator = Accelerator(
        mixed_precision="no" if args.fp32 else "bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(broadcast_buffers=False),
            InitProcessGroupKwargs(timeout=timedelta(minutes=30)),
        ],
    )
    if args.strict_main_budget and accelerator.num_processes != 1:
        raise ValueError(
            "strict exact-step accounting currently requires --num_processes 1"
        )
    random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)
    model_dtype = torch.float32 if args.fp32 else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, token=os.getenv("HF_TOKEN") or None)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, dtype=model_dtype, token=os.getenv("HF_TOKEN") or None
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
        # Keep LoRA weights and Adam moments in FP32.  With the BF16 base
        # model, BF16 adapter gradients became non-finite on the first update
        # carrying a non-zero GRPO advantage.
        for parameter in base_model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    else:
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.model_revision, dtype=model_dtype, token=os.getenv("HF_TOKEN") or None
        ).to(accelerator.device)
        reference_model.config.pad_token_id = tokenizer.pad_token_id
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((parameter for parameter in base_model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    model, optimizer = accelerator.prepare(base_model, optimizer)
    deployable_harness = None
    if args.method in DEPLOYABLE_METHODS:
        generator_model = args.harness_generator_model or args.model
        correction_generator = FrozenModelCorrectionGenerator(
            generator_model,
            revision=args.model_revision,
            device=str(accelerator.device),
            max_new_tokens=64,
        )
        deployable_harness = DeployableAPIBankHarness("H3", correction_generator=correction_generator)
    budget = BudgetCounter()
    audit_ledger = AuditLedger(args.output_dir / f"audit_rank{accelerator.process_index}.jsonl")
    engine = RescueCreditEngine(
        PatchEMA(beta=0.95),
        UniformAuditScheduler(
            args.audit_probability,
            warm_start_events_per_patch=args.audit_warm_start_events,
        ),
        audit_ledger,
        budget,
        exact_confidence_threshold=2.0 if args.force_shadow_credit else 0.99,
    )
    event_logger = JsonlLogger(args.output_dir / f"rescue_events_rank{accelerator.process_index}.jsonl")
    preference_logger = JsonlLogger(args.output_dir / f"preference_events_rank{accelerator.process_index}.jsonl")
    train_logger = JsonlLogger(args.output_dir / f"train_rank{accelerator.process_index}.jsonl")
    global_counts = torch.zeros(3, dtype=torch.long, device=accelerator.device)
    global_diagnostic_counts = torch.zeros(6, dtype=torch.long, device=accelerator.device)
    local_curriculum_assignments = 0
    local_task_assignments = 0
    local_trained_updates = 0
    local_discarded_partial_group_candidates = 0
    curriculum_strata = ("unique_visible_identifier", "visible_tool_dependency")
    local_stratum_assignments = {stratum: 0 for stratum in curriculum_strata}
    curriculum.select_batch(0, accelerator.num_processes)
    start_time = time.time()

    for update in range(args.max_updates):
        max_main_cost = max(task["max_steps"] for task in tasks) * args.group_size * accelerator.num_processes
        max_total_cost = max_main_cost * (2 if args.method in {"rescuecredit", "rescuecredit_v2", "full_shadow"} else 1)
        if should_stop_for_budget(
            args.budget_mode,
            int(global_counts[0].item()),
            int(global_counts.sum().item()),
            args.main_interaction_budget,
            args.total_interaction_budget,
            max_total_cost,
        ):
            break
        curriculum_selection = curriculum.select_batch(update, accelerator.num_processes)[
            accelerator.process_index
        ]
        task = curriculum_selection.task
        local_task_assignments += 1
        local_curriculum_assignments += int(curriculum_selection.source == "visible_curriculum")
        if curriculum_selection.stratum is not None:
            local_stratum_assignments[curriculum_selection.stratum] += 1
        before = (budget.main_steps, budget.shadow_steps, budget.failed_replay_steps)
        traces = []
        for candidate in range(args.group_size):
            trace = rollout_candidate(
                model,
                reference_model,
                tokenizer,
                task,
                update,
                candidate,
                args,
                accelerator,
                engine,
                budget,
                event_logger,
                deployable_harness,
                (
                    args.main_interaction_budget
                    if args.strict_main_budget
                    else None
                ),
            )
            if not trace.truncated_for_budget:
                traces.append(trace)
            if (
                args.strict_main_budget
                and budget.main_steps >= args.main_interaction_budget
            ):
                break
        local_delta = torch.tensor(
            [budget.main_steps - before[0], budget.shadow_steps - before[1], budget.failed_replay_steps - before[2]],
            dtype=torch.long,
            device=accelerator.device,
        )
        global_counts += accelerator.reduce(local_delta, reduction="sum")
        local_diagnostic = torch.tensor(
            [
                sum(trace.diagnostic_shadow_steps for trace in traces),
                sum(trace.diagnostic_replay_valid is True for trace in traces),
                sum(
                    trace.diagnostic_replay_valid is True
                    and trace.diagnostic_shadow_return is not None
                    and trace.assisted_return > trace.diagnostic_shadow_return
                    for trace in traces
                ),
                sum(
                    trace.diagnostic_replay_valid is True
                    and trace.diagnostic_shadow_return is not None
                    and trace.assisted_return < trace.diagnostic_shadow_return
                    for trace in traces
                ),
                sum(
                    trace.diagnostic_replay_valid is True
                    and trace.diagnostic_shadow_return is not None
                    and abs(trace.assisted_return - trace.diagnostic_shadow_return) <= 1e-12
                    for trace in traces
                ),
                sum(
                    trace.diagnostic_shadow_steps
                    for trace in traces
                    if not trace.diagnostic_shadow_already_charged
                ),
            ],
            dtype=torch.long,
            device=accelerator.device,
        )
        global_diagnostic_counts += accelerator.reduce(
            local_diagnostic, reduction="sum"
        )
        if not traces:
            break
        if args.strict_main_budget and len(traces) != args.group_size:
            # Never normalize or optimize a partial GRPO group. Its complete
            # trajectories remain part of the exact interaction-cost ledger,
            # but not of the learning dataset.
            local_discarded_partial_group_candidates += len(traces)
            break
        local_trained_updates += 1
        gh_advantages = group_normalize([trace.assisted_return for trace in traces])
        g0_advantages = group_normalize([trace.g0_hat for trace in traces])
        invalid_policy_steps = 0

        for policy_epoch in range(args.policy_epochs):
            model.eval()
            term_specs = [
                (candidate_index, trace, step)
                for candidate_index, trace in enumerate(traces)
                for step in trace.steps
            ]
            if not term_specs:
                raise RuntimeError("rollout produced no trainable policy steps")
            optimizer.zero_grad(set_to_none=True)
            grpo_value = 0.0
            kl_value = 0.0
            loss_corr_value = 0.0
            loss_causal_value = 0.0
            sync_context = model.no_sync() if hasattr(model, "no_sync") else contextlib.nullcontext()
            # Variable-length agent trajectories produce a different number of
            # forwards on each rank. Accumulate and free one graph at a time,
            # then execute a rank-uniform manual gradient reduction below.
            with sync_context:
                for candidate_index, trace, step in term_specs:
                    ids = step.input_ids.to(accelerator.device)
                    current = completion_token_logprobs(model, ids, step.prompt_length)
                    old = step.old_logprobs.to(accelerator.device)
                    reference = step.reference_logprobs.to(accelerator.device)
                    if (
                        current.numel() == 0
                        or current.shape != old.shape
                        or current.shape != reference.shape
                        or not torch.isfinite(current).all().item()
                        or not torch.isfinite(old).all().item()
                        or not torch.isfinite(reference).all().item()
                    ):
                        invalid_policy_steps += 1
                        continue
                    advantage = routed_advantage(
                        args.method,
                        trace.event_step,
                        step.step_index,
                        gh_advantages[candidate_index],
                        g0_advantages[candidate_index],
                    )
                    log_importance_ratio = (current.float() - old.float()).clamp(-20.0, 20.0)
                    ratio = torch.exp(log_importance_ratio)
                    scalar = torch.tensor(advantage, dtype=current.dtype, device=current.device)
                    unclipped = ratio * scalar
                    clipped = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps) * scalar
                    step_grpo = -torch.minimum(unclipped, clipped).mean()
                    log_ratio = (reference.float() - current.float()).clamp(-10.0, 10.0)
                    # The k2 approximation is non-negative and has a bounded
                    # derivative after clamping.  The previous exponential k3
                    # estimator could overflow activation gradients before
                    # global gradient clipping had a chance to run.
                    step_kl = 0.5 * log_ratio.square().mean()
                    accelerator.backward((step_grpo + args.kl_coef * step_kl) / len(term_specs))
                    grpo_value += float(step_grpo.detach())
                    kl_value += float(step_kl.detach())
                preferences = [trace.preference for trace in traces if trace.preference is not None]
                if preferences:
                    for prompt, chosen, rejected in preferences:
                        chosen_logp = conditional_logprob(
                            model, tokenizer, [prompt], [chosen], accelerator.device
                        )
                        rejected_logp = conditional_logprob(
                            model, tokenizer, [prompt], [rejected], accelerator.device
                        )
                        preference_loss = -torch.nn.functional.logsigmoid(chosen_logp - rejected_logp).mean()
                        accelerator.backward(args.lambda_corr * preference_loss / len(preferences))
                        loss_corr_value += float(preference_loss.detach()) / len(preferences)
                v2_preferences = [trace.v2_preference for trace in traces if trace.v2_preference is not None]
                if v2_preferences:
                    for sample in v2_preferences:
                        logp_a = conditional_logprob(
                            model, tokenizer, [sample.prompt], [sample.action_a], accelerator.device
                        )
                        logp_b = conditional_logprob(
                            model, tokenizer, [sample.prompt], [sample.action_b], accelerator.device
                        )
                        margin = args.preference_beta * (logp_b - logp_a)
                        zero = margin.sum() * 0.0
                        ordinary_loss = zero
                        if sample.decision.ordinary_direction == "b_over_a":
                            ordinary_loss = torch.nn.functional.softplus(-margin).mean()
                        elif sample.decision.ordinary_direction == "a_over_b":
                            ordinary_loss = torch.nn.functional.softplus(margin).mean()
                        causal_loss = zero
                        if sample.decision.causal_direction == "b_over_a":
                            causal_loss = torch.nn.functional.softplus(-margin).mean()
                        elif sample.decision.causal_direction == "a_over_b":
                            causal_loss = torch.nn.functional.softplus(margin).mean()
                        weighted_causal_loss = sample.decision.causal_weight * causal_loss
                        combined = args.lambda_corr * ordinary_loss + args.lambda_causal * weighted_causal_loss
                        accelerator.backward(combined / len(v2_preferences))
                        ordinary_value = float(ordinary_loss.detach())
                        causal_value = float(weighted_causal_loss.detach())
                        loss_corr_value += ordinary_value / len(v2_preferences)
                        loss_causal_value += causal_value / len(v2_preferences)
                        preference_logger.write(
                            {
                                "update": update,
                                "policy_epoch": policy_epoch,
                                "rank": accelerator.process_index,
                                "task_id": task["task_id"],
                                "episode_id": sample.event.episode_id,
                                "candidate_id": sample.event.candidate_id,
                                "patch_id": sample.event.patch_id,
                                "a_executable_valid": sample.event.metadata.get("a_executable_valid"),
                                "b_executable_valid": sample.event.metadata.get("b_executable_valid"),
                                "a_semantic_valid": sample.event.metadata.get("a_semantic_valid"),
                                "b_semantic_valid": sample.event.metadata.get("b_semantic_valid"),
                                "assisted_return": sample.event.assisted_return,
                                "shadow_return": sample.event.shadow_return,
                                "delta": sample.decision.delta,
                                "audit_draw": sample.event.audit_draw,
                                "audit_probability": sample.decision.audit_probability,
                                "causal_decision": sample.decision.causal_decision,
                                "causal_weight": sample.decision.causal_weight,
                                "ordinary_direction": sample.decision.ordinary_direction,
                                "causal_direction": sample.decision.causal_direction,
                                "preference_margin": float((logp_b - logp_a).detach().mean()),
                                "correction_loss": ordinary_value,
                                "causal_loss": causal_value,
                                "length_normalization": "mean_action_token_logprob",
                            }
                        )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                for parameter in model.parameters():
                    if not parameter.requires_grad:
                        continue
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                    parameter.grad.div_(accelerator.num_processes)
            loss_grpo = torch.tensor(grpo_value / len(term_specs), device=accelerator.device)
            loss_kl = torch.tensor(kl_value / len(term_specs), device=accelerator.device)
            loss_corr = torch.tensor(loss_corr_value, device=accelerator.device)
            loss_causal = torch.tensor(loss_causal_value, device=accelerator.device)
            loss = (
                loss_grpo
                + args.kl_coef * loss_kl
                + args.lambda_corr * loss_corr
                + args.lambda_causal * loss_causal
            )
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at update {update}: {grad_norm.item()}")
            optimizer.step()

        train_logger.write(
            {
                "update": update,
                "rank": accelerator.process_index,
                "method": args.method,
                "run_id": args.run_id,
                "task_id": task["task_id"],
                "sampler_source": curriculum_selection.source,
                "curriculum_stratum": curriculum_selection.stratum,
                "visible_structure_reasons": list(curriculum_selection.reasons),
                "config_hash": config_hash,
                "global_main_steps": int(global_counts[0].item()),
                "global_shadow_steps": int(global_counts[1].item()),
                "global_failed_replay_steps": int(global_counts[2].item()),
                "global_training_steps": int(global_counts.sum().item()),
                "loss": float(loss.detach()),
                "loss_grpo": float(loss_grpo.detach()),
                "loss_kl": float(loss_kl.detach()),
                "loss_corr": float(loss_corr.detach()),
                "weighted_loss_corr": float(
                    (args.lambda_corr * loss_corr).detach()
                ),
                "loss_causal": float(loss_causal.detach()),
                "invalid_policy_steps": invalid_policy_steps,
                "assisted_returns": [trace.assisted_return for trace in traces],
                "realized_group_size": len(traces),
                "g0_hat": [trace.g0_hat for trace in traces],
                "diagnostic_shadow_returns": [
                    trace.diagnostic_shadow_return for trace in traces
                ],
                "diagnostic_replay_valid": [
                    trace.diagnostic_replay_valid for trace in traces
                ],
                "prefix_assigned_advantages": [
                    (
                        None
                        if trace.event_step is None
                        else gh_advantages[index]
                        if args.method == "naive_h_grpo"
                        else 0.0
                        if args.method == "mask_correction"
                        else g0_advantages[index]
                    )
                    for index, trace in enumerate(traces)
                ],
            }
        )
        if accelerator.is_main_process and (update + 1) % args.save_every == 0:
            checkpoint = args.output_dir / "checkpoints" / f"update_{update + 1}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(checkpoint, save_function=accelerator.save)
            tokenizer.save_pretrained(checkpoint)

    audit_stats = accelerator.reduce(
        torch.tensor(
            [
                engine.eligible_events,
                engine.audited_events,
                engine.valid_audits,
                engine.scheduler.warm_start_assignments,
            ],
            dtype=torch.long,
            device=accelerator.device,
        ),
        reduction="sum",
    )
    sampler_stats = accelerator.reduce(
        torch.tensor(
            [
                local_curriculum_assignments,
                local_task_assignments,
                *(local_stratum_assignments[stratum] for stratum in curriculum_strata),
                local_trained_updates,
                local_discarded_partial_group_candidates,
            ],
            dtype=torch.long,
            device=accelerator.device,
        ),
        reduction="sum",
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = args.output_dir / "checkpoints" / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(final_dir)
        checkpoint_sha256 = directory_sha256(final_dir)
        training_log_artifacts = []
        for training_log in sorted(args.output_dir.glob("train_rank*.jsonl")):
            training_log_artifacts.append(
                {
                    "path": training_log.name,
                    "sha256": file_sha256(training_log),
                    "rows": sum(
                        1
                        for line in training_log.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if line.strip()
                    ),
                }
            )
        budget_target = (
            args.main_interaction_budget if args.budget_mode == "main" else args.total_interaction_budget
        )
        budget_used = int(global_counts[0].item()) if args.budget_mode == "main" else int(global_counts.sum().item())
        total_task_assignments = int(sampler_stats[1].item())
        if total_task_assignments % accelerator.num_processes:
            raise RuntimeError("global task assignments are not divisible by world size")
        completed_updates = total_task_assignments // accelerator.num_processes
        write_json(
            args.output_dir / "run_summary.json",
            {
                "method": args.method,
                "run_id": args.run_id,
                "seed": args.seed,
                "model": args.model,
                "model_revision": args.model_revision,
                "split_hash": manifest["split_hashes"]["train"],
                "config_hash": config_hash,
                "experiment_protocol_lock_sha256": protocol_lock_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "training_logs": training_log_artifacts,
                "world_size": accelerator.num_processes,
                "main_steps": int(global_counts[0].item()),
                "shadow_steps": int(global_counts[1].item()),
                "failed_replay_steps": int(global_counts[2].item()),
                "total_training_steps": int(global_counts.sum().item()),
                "interaction_budget": args.total_interaction_budget,
                "budget_mode": args.budget_mode,
                "main_interaction_budget": args.main_interaction_budget or None,
                "budget_target": budget_target,
                "budget_unused": max(0, budget_target - budget_used),
                "budget_overshoot": max(0, budget_used - budget_target),
                "shadow_steps_reported_as_extra_cost": args.budget_mode == "main",
                "sampling": {
                    "mode": (
                        "visible_structure_mix"
                        if args.visible_curriculum_fraction > 0.0
                        else "sequential"
                    ),
                    "visible_curriculum_fraction": args.visible_curriculum_fraction,
                    "visible_pool_size": curriculum.visible_pool_size,
                    "total_pool_size": len(tasks),
                    "visible_pool_hash": curriculum.visible_pool_hash,
                    "visible_reason_pool_sizes": curriculum.reason_pool_sizes,
                    "reference_free_selection": curriculum.reference_free_selection,
                    "curriculum_assignments": int(sampler_stats[0].item()),
                    "total_task_assignments": total_task_assignments,
                    "realized_visible_fraction": (
                        int(sampler_stats[0].item()) / max(1, total_task_assignments)
                    ),
                    "curriculum_stratum_assignments": {
                        stratum: int(sampler_stats[index + 2].item())
                        for index, stratum in enumerate(curriculum_strata)
                    },
                    "trained_updates": int(sampler_stats[4].item()),
                    "discarded_partial_group_candidates": int(
                        sampler_stats[5].item()
                    ),
                    "assignment_sequence_hash": curriculum.assignment_sequence_hash(
                        completed_updates,
                        accelerator.num_processes,
                    ),
                },
                "audit_stats": {
                    "eligible_events": int(audit_stats[0].item()),
                    "audited_events": int(audit_stats[1].item()),
                    "valid_audits": int(audit_stats[2].item()),
                    "warm_start_assignments": int(audit_stats[3].item()),
                },
                "diagnostic_full_shadow": {
                    "enabled": args.diagnostic_full_shadow,
                    "steps": int(global_diagnostic_counts[0].item()),
                    "replay_valid_events": int(global_diagnostic_counts[1].item()),
                    "rescue_events": int(global_diagnostic_counts[2].item()),
                    "harm_events": int(global_diagnostic_counts[3].item()),
                    "zero_delta_events": int(global_diagnostic_counts[4].item()),
                    "unique_extra_steps": int(global_diagnostic_counts[5].item()),
                    "steps_already_counted_in_training_or_failed_replay": int(
                        global_diagnostic_counts[0].item()
                        - global_diagnostic_counts[5].item()
                    ),
                    "used_for_naive_or_mask_training": False,
                },
                "authoritative_unique_interaction_steps": int(
                    global_counts.sum().item() + global_diagnostic_counts[5].item()
                ),
                "comparability": {
                    "group_size": args.group_size,
                    "max_updates": args.max_updates,
                    "save_every": args.save_every,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "max_shadow_steps": args.max_shadow_steps,
                    "interaction_budget": args.total_interaction_budget,
                    "budget_mode": args.budget_mode,
                    "main_interaction_budget": args.main_interaction_budget or None,
                    "learning_rate": args.learning_rate,
                    "clip_eps": args.clip_eps,
                    "policy_epochs": args.policy_epochs,
                    "kl_coef": args.kl_coef,
                    "lambda_corr": args.lambda_corr,
                    "lambda_causal": args.lambda_causal,
                    "preference_beta": args.preference_beta,
                    "max_causal_weight": args.max_causal_weight,
                    "audit_probability": args.audit_probability,
                    "audit_warm_start_events": args.audit_warm_start_events,
                    "visible_curriculum_fraction": args.visible_curriculum_fraction,
                    "use_lora": args.use_lora,
                    "fp32": args.fp32,
                    "diagnostic_full_shadow": args.diagnostic_full_shadow,
                    "force_shadow_credit": args.force_shadow_credit,
                    "strict_main_budget": args.strict_main_budget,
                    "harness_mode": (
                        "deployable"
                        if args.method in DEPLOYABLE_METHODS
                        else "oracle_teacher"
                    ),
                    "harness_generator_model": (
                        args.harness_generator_model or args.model
                        if args.method in DEPLOYABLE_METHODS
                        else None
                    ),
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
