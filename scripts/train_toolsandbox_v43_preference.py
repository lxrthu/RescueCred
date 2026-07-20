#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import JsonlLogger, write_json
from rescuecredit.toolsandbox_preference import canonical_completion, training_preference
from scripts.train_route_a_preference import (
    balanced_causal_epoch_order,
    mean_completion_logprob,
    preference_loss_components,
)


PROTOCOL_STATUS = "frozen_before_toolsandbox_v43_training"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("mask", "v43"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--absolute-margin-coef", type=float, default=1.0)
    parser.add_argument("--target-margin", type=float, default=0.05)
    parser.add_argument("--reference-anchor-coef", type=float, default=0.25)
    parser.add_argument("--presentations-per-epoch", type=int, default=60)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("invalid ToolSandbox V4.3 protocol status")
    if args.method not in protocol.get("methods", []):
        raise ValueError("method is absent from the frozen V4.3 comparison")
    config = {
        "seed": args.seed,
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
        "preference_weight": "unit_direction",
        "sampling": "identical_multi_prefix_class_balanced",
    }
    if config != protocol.get("config"):
        raise ValueError("training CLI does not match the frozen V4.3 config")
    if file_sha256(args.train_file) != protocol.get("train_sha256"):
        raise ValueError("training file does not match the frozen protocol")
    base_model_sha256 = directory_sha256(args.model)
    if base_model_sha256 != protocol.get("base_model_sha256"):
        raise ValueError("base model does not match the frozen protocol")
    source_hashes = protocol.get("source_sha256", {})
    if not source_hashes or not all(
        Path(path).is_file() and file_sha256(Path(path)) == expected
        for path, expected in source_hashes.items()
    ):
        raise ValueError("V4.3 source identity changed after protocol freeze")

    rows = read_jsonl(args.train_file)
    if len(rows) != int(protocol["train_events"]):
        raise ValueError("training row count does not match the frozen protocol")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(args.output_dir / "train.jsonl")
    started = time.time()
    optimizer_steps = 0
    presentations = 0
    accumulated = 0
    reference_margin_cache: dict[str, float] = {}
    presented_decisions: Counter[str] = Counter()
    presented_source_decisions: Counter[str] = Counter()
    presented_event_ids: list[str] = []
    optimizer.zero_grad(set_to_none=True)
    preference_method = "mask" if args.method == "mask" else "v4"
    for epoch in range(args.epochs):
        epoch_rows = balanced_causal_epoch_order(
            rows, args.seed, epoch, args.presentations_per_epoch
        )
        for row in epoch_rows:
            chosen, rejected, weight = training_preference(row, preference_method)
            prompt = str(row["prompt"])
            chosen_text = canonical_completion(chosen)
            rejected_text = canonical_completion(rejected)
            model.train()
            policy_chosen = mean_completion_logprob(
                model, tokenizer, prompt, chosen_text, args.max_length, device
            )
            policy_rejected = mean_completion_logprob(
                model, tokenizer, prompt, rejected_text, args.max_length, device
            )
            policy_margin = policy_chosen - policy_rejected
            cache_key = f"{row['event_id']}:{chosen_text}>{rejected_text}"
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
            base_loss, dpo_loss, absolute_loss = preference_loss_components(
                policy_margin,
                reference_margin,
                weight=weight,
                beta=args.beta,
                absolute_margin_coef=args.absolute_margin_coef,
                target_margin=args.target_margin,
            )
            anchor_loss = (policy_margin - reference_margin).square()
            loss = base_loss + args.reference_anchor_coef * anchor_loss
            (loss / args.gradient_accumulation).backward()
            accumulated += 1
            presentations += 1
            presented_event_ids.append(str(row["event_id"]))
            presented_source_decisions[str(row["decision"])] += 1
            label = "b_over_a" if chosen == row["action_b"] else "a_over_b"
            presented_decisions[label] += 1
            logger.write(
                {
                    "epoch": epoch,
                    "event_id": row["event_id"],
                    "source_decision": row["decision"],
                    "training_label": label,
                    "loss": float(loss.detach()),
                    "loss_dpo": float(dpo_loss.detach()),
                    "loss_absolute_margin": float(absolute_loss.detach()),
                    "loss_reference_anchor": float(anchor_loss.detach()),
                    "policy_margin": float(policy_margin.detach()),
                    "reference_margin": float(reference_margin.detach()),
                }
            )
            if accumulated == args.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated = 0
    if accumulated:
        scale = args.gradient_accumulation / accumulated
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer_steps += 1

    sequence_sha256 = hashlib.sha256(
        "\n".join(presented_event_ids).encode("utf-8")
    ).hexdigest()
    if sequence_sha256 != protocol.get("expected_presented_event_sequence_sha256"):
        raise RuntimeError("actual event sequence differs from the frozen protocol")
    expected_labels = protocol.get("expected_presented_decisions", {}).get(args.method)
    if dict(sorted(presented_decisions.items())) != expected_labels:
        raise RuntimeError("actual training labels differ from the frozen protocol")

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_v43_multi_prefix_anchored_training",
        "method": args.method,
        "model": str(args.model),
        "base_model_sha256": base_model_sha256,
        "adapter": str(adapter_dir),
        "adapter_sha256": directory_sha256(adapter_dir),
        "protocol_lock": str(args.protocol_lock),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "train_file_sha256": file_sha256(args.train_file),
        "train_events": len(rows),
        **config,
        "active_event_presentations": presentations,
        "presented_source_decisions": dict(sorted(presented_source_decisions.items())),
        "presented_decisions": dict(sorted(presented_decisions.items())),
        "presented_event_sequence_sha256": sequence_sha256,
        "optimizer_steps": optimizer_steps,
        "cached_reference_margins": len(reference_margin_cache),
        "same_data_same_budget_role": True,
        "loss_definition": (
            "unit_weight*(dpo_shift+absolute_margin)+reference_anchor"
        ),
        "scope": "offline ToolSandbox preference learning; not autonomous task success",
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
