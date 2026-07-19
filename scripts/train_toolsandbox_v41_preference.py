#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import JsonlLogger, write_json
from rescuecredit.toolsandbox_preference import (
    canonical_completion,
    matched_epoch_order,
    training_preference,
)
from scripts.train_route_a_preference import (
    mean_completion_logprob,
    preference_loss_components,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("mask", "v4"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_toolsandbox_v41_preference_outcomes":
        raise ValueError("invalid ToolSandbox V4.1 preference protocol status")
    if args.method not in protocol.get("methods", []):
        raise ValueError("method is absent from the frozen comparison")
    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "fp32": args.fp32,
        "preference_weight": "unit_direction",
        "sampling": "identical_natural_event_order",
    }
    if config != protocol.get("config"):
        raise ValueError("training CLI does not match the frozen V4.1 config")
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
        raise ValueError("V4.1 preference source identity changed after freeze")

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
    presented_event_ids: list[str] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for row in matched_epoch_order(rows, args.seed, epoch):
            chosen, rejected, weight = training_preference(row, args.method)
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
            loss, dpo_loss, _ = preference_loss_components(
                policy_margin,
                reference_margin,
                weight=weight,
                beta=args.beta,
                absolute_margin_coef=0.0,
                target_margin=0.0,
            )
            (loss / args.gradient_accumulation).backward()
            accumulated += 1
            presentations += 1
            presented_event_ids.append(str(row["event_id"]))
            label = (
                "b_over_a"
                if chosen == row["action_b"]
                else "a_over_b"
            )
            presented_decisions[label] += 1
            logger.write(
                {
                    "epoch": epoch,
                    "event_id": row["event_id"],
                    "source_decision": row["decision"],
                    "training_label": label,
                    "loss": float(loss.detach()),
                    "loss_dpo": float(dpo_loss.detach()),
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

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_v41_preference_training",
        "method": args.method,
        "seed": args.seed,
        "model": str(args.model),
        "base_model_sha256": base_model_sha256,
        "adapter": str(adapter_dir),
        "adapter_sha256": directory_sha256(adapter_dir),
        "protocol_lock": str(args.protocol_lock),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "train_file_sha256": file_sha256(args.train_file),
        "train_events": len(rows),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "fp32": args.fp32,
        "presentations_per_epoch": len(rows),
        "active_event_presentations": presentations,
        "presented_decisions": dict(sorted(presented_decisions.items())),
        "presented_event_sequence_sha256": __import__("hashlib").sha256(
            "\n".join(presented_event_ids).encode("utf-8")
        ).hexdigest(),
        "optimizer_steps": optimizer_steps,
        "cached_reference_margins": len(reference_margin_cache),
        "same_data_same_budget_role": True,
        "preference_weight": "unit_direction",
        "scope": "offline ToolSandbox preference learning; not autonomous task success",
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
