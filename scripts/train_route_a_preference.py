#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import time
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import JsonlLogger, write_json
from rescuecredit.route_a_preference import (
    completion,
    epoch_order,
    preference_kind,
    training_preference,
)


def balanced_causal_epoch_order(
    rows: list[dict[str, Any]],
    seed: int,
    epoch: int,
    target_presentations: int,
) -> list[dict[str, Any]]:
    """Build a deterministic, class-balanced V2 epoch with a matched budget."""

    if target_presentations < 2:
        raise ValueError("target_presentations must be at least 2")
    groups = {
        decision: [row for row in rows if row.get("decision") == decision]
        for decision in ("rescue_preference", "reverse_preference")
    }
    missing = [decision for decision, group in groups.items() if not group]
    if missing:
        raise ValueError(f"missing causal training class(es): {missing}")

    counts = {
        "rescue_preference": target_presentations // 2,
        "reverse_preference": target_presentations // 2,
    }
    # Alternate the odd presentation across epochs to avoid a persistent bias.
    if target_presentations % 2:
        extra = ("rescue_preference", "reverse_preference")[epoch % 2]
        counts[extra] += 1

    sampled: list[dict[str, Any]] = []
    for group_index, (decision, group) in enumerate(groups.items()):
        ordered = list(group)
        random.Random(seed + epoch * 1009 + group_index * 104729).shuffle(ordered)
        offset = (epoch * counts[decision]) % len(ordered)
        sampled.extend(
            ordered[(offset + index) % len(ordered)]
            for index in range(counts[decision])
        )
    random.Random(seed + epoch * 1009 + 99991).shuffle(sampled)
    return sampled


def validity_first_epoch_order(
    rows: list[dict[str, Any]],
    seed: int,
    epoch: int,
    target_presentations: int,
) -> list[dict[str, Any]]:
    """Natural-ratio V3.1 sampling after the public validity gate."""

    teachable = [
        row for row in rows if training_preference(row, "v31") is not None
    ]
    if not teachable:
        raise ValueError("V3.1 has no validity-gated teachable rows")
    ordered = list(teachable)
    random.Random(seed + epoch * 1009 + 314159).shuffle(ordered)
    offset = (epoch * target_presentations) % len(ordered)
    sampled = [
        ordered[(offset + index) % len(ordered)]
        for index in range(target_presentations)
    ]
    random.Random(seed + epoch * 1009 + 271828).shuffle(sampled)
    return sampled


def preference_loss_components(
    policy_margin,
    reference_margin,
    *,
    weight: float,
    beta: float,
    absolute_margin_coef: float,
    target_margin: float,
):
    """DPO shift loss plus an optional absolute chosen-over-rejected margin."""

    import torch

    dpo_loss = torch.nn.functional.softplus(
        -float(beta) * (policy_margin - reference_margin)
    )
    absolute_loss = torch.nn.functional.softplus(
        float(beta) * (float(target_margin) - policy_margin)
    )
    total = float(weight) * (
        dpo_loss + float(absolute_margin_coef) * absolute_loss
    )
    return total, dpo_loss, absolute_loss


def mean_completion_logprob(model, tokenizer, prompt: str, completion_text: str, max_length: int, device):
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    action_ids = tokenizer(completion_text, add_special_tokens=False).input_ids
    if not action_ids:
        raise ValueError("preference completion has no tokens")
    if len(action_ids) >= max_length:
        action_ids = action_ids[: max_length - 1]
    prompt_budget = max(1, max_length - len(action_ids))
    prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = torch.tensor([prompt_ids + action_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    start = len(prompt_ids) - 1
    selected = token_logprobs[:, start : start + len(action_ids)]
    return selected.mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=["mask", "v2", "v3", "v31"], required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-causal-weight", type=float, default=2.5)
    parser.add_argument("--absolute-margin-coef", type=float, default=0.0)
    parser.add_argument("--target-margin", type=float, default=0.0)
    parser.add_argument(
        "--v2-presentations-per-epoch",
        type=int,
        default=0,
        help="0 matches the Mask presentation budget (number of train rows)",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = read_jsonl(args.train_file)
    if args.absolute_margin_coef < 0:
        raise ValueError("--absolute-margin-coef must be non-negative")
    if args.target_margin < 0:
        raise ValueError("--target-margin must be non-negative")
    if args.method in {"v3", "v31"} and args.absolute_margin_coef <= 0:
        raise ValueError("V3/V3.1 requires --absolute-margin-coef > 0")
    if args.method in {"v3", "v31"} and args.protocol_lock is None:
        raise ValueError("V3/V3.1 requires --protocol-lock")
    if args.method not in {"v3", "v31"} and args.absolute_margin_coef != 0:
        raise ValueError("absolute margin loss is only defined for V3/V3.1")
    protocol_lock_sha256 = None
    if args.protocol_lock is not None:
        protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
        expected_status = (
            "frozen_before_v31_outcomes"
            if args.method == "v31"
            else "frozen_before_v3_outcomes"
        )
        if protocol.get("status") != expected_status:
            raise ValueError("invalid V3/V3.1 protocol lock status")
        if args.method == "v31":
            source_hashes = protocol.get("source_sha256", {})
            if not source_hashes or not all(
                Path(path).is_file() and file_sha256(Path(path)) == expected
                for path, expected in source_hashes.items()
            ):
                raise ValueError("V3.1 source identity changed after protocol freeze")
        actual_config = {
            "method": args.method,
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation": args.gradient_accumulation,
            "max_length": args.max_length,
            "beta": args.beta,
            "max_causal_weight": args.max_causal_weight,
            "v2_presentations_per_epoch": args.v2_presentations_per_epoch,
            "absolute_margin_coef": args.absolute_margin_coef,
            "target_margin": args.target_margin,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "fp32": args.fp32,
        }
        if actual_config != protocol.get("config"):
            raise ValueError(
                "V3 CLI config does not match the frozen protocol: "
                f"actual={actual_config}, frozen={protocol.get('config')}"
            )
        if file_sha256(args.train_file) != protocol.get("train_sha256"):
            raise ValueError("V3 train file does not match the frozen protocol")
        base_model_sha256 = directory_sha256(args.model)
        if base_model_sha256 != protocol.get("base_model_sha256"):
            raise ValueError("V3 base model does not match the frozen protocol")
        protocol_lock_sha256 = file_sha256(args.protocol_lock)
    else:
        base_model_sha256 = directory_sha256(args.model)
    if args.v2_presentations_per_epoch < 0:
        raise ValueError("--v2-presentations-per-epoch must be non-negative")
    presentations_per_epoch = (
        len(rows)
        if args.method == "mask" or args.v2_presentations_per_epoch == 0
        else args.v2_presentations_per_epoch
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.enable_input_require_grads()
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    logger = JsonlLogger(args.output_dir / "train.jsonl")
    started = time.time()
    optimizer_steps = 0
    active_events = 0
    skipped_events = 0
    accumulated = 0
    reference_margin_cache: dict[str, float] = {}
    presented_decisions: Counter[str] = Counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        if args.method == "mask":
            epoch_rows = epoch_order(rows, args.seed, epoch)
        elif args.method == "v31":
            epoch_rows = validity_first_epoch_order(
                rows,
                args.seed,
                epoch,
                presentations_per_epoch,
            )
        else:
            epoch_rows = balanced_causal_epoch_order(
                rows,
                args.seed,
                epoch,
                presentations_per_epoch,
            )
        for row in epoch_rows:
            preference = training_preference(
                row, args.method, max_weight=args.max_causal_weight
            )
            if preference is None:
                skipped_events += 1
                continue
            chosen, rejected, weight = preference
            prompt = str(row["prompt"])
            chosen_text = completion(chosen)
            rejected_text = completion(rejected)
            model.train()
            policy_chosen = mean_completion_logprob(
                model, tokenizer, prompt, chosen_text, args.max_length, device
            )
            policy_rejected = mean_completion_logprob(
                model, tokenizer, prompt, rejected_text, args.max_length, device
            )
            policy_margin = policy_chosen - policy_rejected
            cache_key = f"{row['event_id']}:{completion(chosen)}>{completion(rejected)}"
            if cache_key not in reference_margin_cache:
                with torch.no_grad(), model.disable_adapter():
                    reference_chosen = mean_completion_logprob(
                        model, tokenizer, prompt, chosen_text, args.max_length, device
                    )
                    reference_rejected = mean_completion_logprob(
                        model, tokenizer, prompt, rejected_text, args.max_length, device
                    )
                reference_margin_cache[cache_key] = float(
                    (reference_chosen - reference_rejected).detach()
                )
            reference_margin = torch.tensor(
                reference_margin_cache[cache_key],
                dtype=policy_margin.dtype,
                device=device,
            )
            loss, dpo_loss, absolute_loss = preference_loss_components(
                policy_margin,
                reference_margin,
                weight=weight,
                beta=args.beta,
                absolute_margin_coef=args.absolute_margin_coef,
                target_margin=args.target_margin,
            )
            (loss / args.gradient_accumulation).backward()
            active_events += 1
            presented_decisions[preference_kind(row, args.method)] += 1
            accumulated += 1
            logger.write(
                {
                    "epoch": epoch,
                    "event_id": row["event_id"],
                    "decision": row["decision"],
                    "weight": weight,
                    "loss": float(loss.detach()),
                    "loss_dpo": float(dpo_loss.detach()),
                    "loss_absolute_margin": float(absolute_loss.detach()),
                    "policy_margin": float(policy_margin.detach()),
                    "reference_margin": float(reference_margin.detach()),
                }
            )
            if accumulated >= args.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated = 0
    if accumulated:
        # Earlier complete batches were divided by gradient_accumulation. Restore
        # the same mean-gradient normalization for the final partial batch.
        scale = args.gradient_accumulation / accumulated
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer_steps += 1
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    adapter_sha256 = directory_sha256(adapter_dir)
    summary = {
        "status": "completed",
        "method": args.method,
        "seed": args.seed,
        "model": str(args.model),
        "base_model_sha256": base_model_sha256,
        "train_file_sha256": file_sha256(args.train_file),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "beta": args.beta,
        "max_causal_weight": args.max_causal_weight,
        "v2_presentations_per_epoch": args.v2_presentations_per_epoch,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "presentations_per_epoch": presentations_per_epoch,
        "presentation_budget_matches_mask": presentations_per_epoch == len(rows),
        "presented_decisions": dict(sorted(presented_decisions.items())),
        "zero_delta_rows_excluded_from_causal_loss": args.method in {"v2", "v3"},
        "validity_first": args.method == "v31",
        "causal_class_balanced": args.method in {"v2", "v3"},
        "absolute_margin_coef": args.absolute_margin_coef,
        "target_margin": args.target_margin,
        "loss_definition": (
            "weight*(dpo_shift+absolute_margin_coef*absolute_margin)"
            if args.method in {"v3", "v31"}
            else "weight*dpo_shift"
        ),
        "active_event_presentations": active_events,
        "skipped_event_presentations": skipped_events,
        "optimizer_steps": optimizer_steps,
        "cached_reference_margins": len(reference_margin_cache),
        "adapter": str(adapter_dir),
        "adapter_sha256": adapter_sha256,
        "protocol_lock": str(args.protocol_lock) if args.protocol_lock else None,
        "protocol_lock_sha256": protocol_lock_sha256,
        "fp32": args.fp32,
        "scope": "offline preference pilot; not AppWorld task-success evidence",
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
