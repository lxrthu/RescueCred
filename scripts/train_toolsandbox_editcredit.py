#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rescuecredit.edit_credit import (
    canonical_action_edits,
    edit_comparison_prompt,
    edit_credit_objective,
    edit_value_completion,
    empirical_binary_auc,
    fold_role,
    parse_public_preference_prompt,
)
from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import JsonlLogger, write_json
from rescuecredit.toolsandbox_preference import canonical_completion
from rescuecredit.toolsandbox_preference import event_set_hash
from scripts.freeze_toolsandbox_editcredit_protocol import STATUS
from scripts.train_route_a_preference import (
    balanced_causal_epoch_order,
    mean_completion_logprob,
)


def _validate_protocol(args, protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("status") != STATUS:
        raise ValueError("invalid EditCredit protocol status")
    if args.method not in protocol.get("methods", []):
        raise ValueError("method absent from frozen EditCredit protocol")
    config = {
        "seed": args.seed,
        "folds": args.folds,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "beta": args.beta,
        "absolute_margin_coef": args.absolute_margin_coef,
        "target_margin": args.target_margin,
        "reference_anchor_coef": args.reference_anchor_coef,
        "presentations_per_epoch": args.presentations_per_epoch,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "fp32": args.fp32,
        "rescue_delta": args.rescue_delta,
    }
    if config != protocol.get("config"):
        raise ValueError(f"training config differs from protocol: {config}")
    if not 0 <= args.fold < args.folds:
        raise ValueError("fold index out of range")
    if file_sha256(args.train_file) != protocol.get("train_sha256"):
        raise ValueError("training bank differs from protocol")
    if directory_sha256(args.model) != protocol.get("base_model_sha256"):
        raise ValueError("base model differs from protocol")
    sources = protocol.get("source_sha256", {})
    if not sources or not all(Path(path).is_file() and file_sha256(Path(path)) == expected for path, expected in sources.items()):
        raise ValueError("EditCredit source identity changed after freeze")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("full_action", "editcredit"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--absolute-margin-coef", type=float, default=1.0)
    parser.add_argument("--target-margin", type=float, default=0.05)
    parser.add_argument("--reference-anchor-coef", type=float, default=0.25)
    parser.add_argument("--presentations-per-epoch", type=int, default=126)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--rescue-delta", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    config = _validate_protocol(args, protocol)
    rows = read_jsonl(args.train_file)
    assignment = {str(key): int(value) for key, value in protocol["task_fold_assignment"].items()}
    train_rows = [
        row
        for row in rows
        if fold_role(row, assignment=assignment, test_fold=args.fold, folds=args.folds) == "train"
    ]
    if not train_rows:
        raise RuntimeError("cross-fit training split is empty")

    torch.manual_seed(args.seed + args.fold * 1009)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + args.fold * 1009)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, torch_dtype=dtype
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(args.output_dir / "train.jsonl")
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    accumulated = optimizer_steps = presentations = 0
    reference_cache: dict[str, float] = {}
    decision_counter: Counter[str] = Counter()
    side_labels: list[int] = []
    side_scores: list[float] = []

    for epoch in range(args.epochs):
        epoch_rows = balanced_causal_epoch_order(
            train_rows, args.seed + args.fold * 1009, epoch, args.presentations_per_epoch
        )
        class_index: Counter[str] = Counter()
        for row in epoch_rows:
            decision = str(row["decision"])
            swap = bool((class_index[decision] + epoch) % 2)
            class_index[decision] += 1
            prompt = str(row["prompt"])
            action_a, action_b = row["action_a"], row["action_b"]
            model.train()
            margins = []
            reference_margins = []
            edit_paths: list[str] = []
            if args.method == "full_action":
                specs = [("/action", prompt, canonical_completion(action_a), canonical_completion(action_b))]
            else:
                payload = parse_public_preference_prompt(prompt)
                specs = [
                    (
                        edit.path,
                        edit_comparison_prompt(public_payload=payload, edit=edit, swap_candidates=swap),
                        edit_value_completion(edit.value_a),
                        edit_value_completion(edit.value_b),
                    )
                    for edit in canonical_action_edits(action_a, action_b)
                ]
            for path, local_prompt, completion_a, completion_b in specs:
                logp_a = mean_completion_logprob(
                    model, tokenizer, local_prompt, completion_a, args.max_length, device
                )
                logp_b = mean_completion_logprob(
                    model, tokenizer, local_prompt, completion_b, args.max_length, device
                )
                margin = logp_b - logp_a
                cache_key = json.dumps(
                    [row["event_id"], path, swap, completion_a, completion_b],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if cache_key not in reference_cache:
                    with torch.no_grad(), model.disable_adapter():
                        ref_a = mean_completion_logprob(
                            model, tokenizer, local_prompt, completion_a, args.max_length, device
                        )
                        ref_b = mean_completion_logprob(
                            model, tokenizer, local_prompt, completion_b, args.max_length, device
                        )
                    reference_cache[cache_key] = float((ref_b - ref_a).detach())
                margins.append(margin)
                reference_margins.append(
                    torch.tensor(reference_cache[cache_key], dtype=margin.dtype, device=device)
                )
                edit_paths.append(path)
            policy_margin = torch.stack(margins).mean()
            reference_margin = torch.stack(reference_margins).mean()
            loss, dpo_loss, absolute_loss, anchor_loss = edit_credit_objective(
                policy_margin,
                reference_margin,
                decision=decision,
                beta=args.beta,
                absolute_margin_coef=args.absolute_margin_coef,
                target_margin=args.target_margin,
                reference_anchor_coef=args.reference_anchor_coef,
            )
            (loss / args.gradient_accumulation).backward()
            accumulated += 1
            presentations += 1
            decision_counter[decision] += 1
            if args.method == "editcredit":
                # Source-only audit: whether B appeared on the right is fixed
                # before the outcome label and counterbalanced within class.
                side_labels.append(1 if decision == "rescue_preference" else 0)
                side_scores.append(0.0 if swap else 1.0)
            logger.write(
                {
                    "epoch": epoch,
                    "fold": args.fold,
                    "method": args.method,
                    "event_id": row["event_id"],
                    "decision": decision,
                    "swap_candidates": swap if args.method == "editcredit" else None,
                    "edit_paths": edit_paths,
                    "loss": float(loss.detach()),
                    "loss_dpo": float(dpo_loss.detach()),
                    "loss_absolute_margin": float(absolute_loss.detach()),
                    "loss_reference_anchor": float(anchor_loss.detach()),
                    "policy_margin_b_over_a": float(policy_margin.detach()),
                    "reference_margin_b_over_a": float(reference_margin.detach()),
                }
            )
            if accumulated == args.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                optimizer_steps += 1
    if accumulated:
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
    source_auc = empirical_binary_auc(side_labels, side_scores) if side_labels else None
    summary = {
        "status": "completed",
        "stage": "toolsandbox_editcredit_crossfit_training",
        "method": args.method,
        "fold": args.fold,
        "model": str(args.model),
        "base_model_sha256": directory_sha256(args.model),
        "adapter": str(adapter_dir),
        "adapter_sha256": directory_sha256(adapter_dir),
        "protocol_lock": str(args.protocol_lock),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "train_file_sha256": file_sha256(args.train_file),
        "train_events": len(train_rows),
        "train_task_groups": len({str(row["task_id_hash"]) for row in train_rows}),
        **config,
        "presentations": presentations,
        "presented_decisions": dict(sorted(decision_counter.items())),
        "optimizer_steps": optimizer_steps,
        "reference_cache_entries": len(reference_cache),
        "presentation_side_label_auc": source_auc,
        "train_event_set_hash": event_set_hash(train_rows),
        "train_task_group_ids": sorted({str(row["task_id_hash"]) for row in train_rows}),
        "source_identity_in_model_input": False if args.method == "editcredit" else True,
        "loss_definition": "signed_counterfactual_dpo_shift_plus_absolute_margin_and_reference_anchor",
        "claim_boundary": protocol["claim_boundary"],
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
