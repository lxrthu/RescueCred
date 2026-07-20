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
from rescuecredit.toolsandbox_preference import (
    canonical_completion,
    matched_epoch_order,
)
from scripts.train_route_a_preference import mean_completion_logprob
from scripts.train_toolsandbox_v43_preference import V46_PROTOCOL_STATUS


def residual_route(
    decision: str, initial_margin: float, confidence_margin: float
) -> tuple[str, float]:
    """Return the selective route and desired residual direction.

    Positive direction increases B over A; negative direction decreases it.
    A correctly signed Mask decision with enough margin is retained.
    """

    if decision == "rescue_preference":
        direction = 1.0
    elif decision == "reverse_preference":
        direction = -1.0
    else:
        raise ValueError(f"unsupported causal decision: {decision}")
    signed_initial = direction * initial_margin
    route = "correct" if signed_initial < confidence_margin else "preserve"
    return route, direction


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("control", "v46"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--mask-adapter", type=Path, required=True)
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--protocol-lock", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=3e-6)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--target-residual", type=float, default=0.05)
    p.add_argument("--confidence-margin", type=float, default=0.05)
    p.add_argument("--retention-coef", type=float, default=1.0)
    p.add_argument("--reference-anchor-coef", type=float, default=0.25)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "beta": args.beta,
        "target_residual": args.target_residual,
        "confidence_margin": args.confidence_margin,
        "retention_coef": args.retention_coef,
        "reference_anchor_coef": args.reference_anchor_coef,
        "fp32": args.fp32,
        "sampling": "all_unique_events_identical_order",
    }
    if protocol.get("status") != V46_PROTOCOL_STATUS or args.method not in protocol.get(
        "methods", []
    ):
        raise ValueError("invalid frozen V4.6 protocol/method")
    if config != protocol.get("config"):
        raise ValueError("V4.6 CLI differs from frozen config")
    if file_sha256(args.train_file) != protocol.get("train_sha256"):
        raise ValueError("V4.6 training data identity mismatch")
    if directory_sha256(args.model) != protocol.get("base_model_sha256"):
        raise ValueError("V4.6 base model identity mismatch")
    if directory_sha256(args.mask_adapter) != protocol.get("mask_adapter_sha256"):
        raise ValueError("V4.6 Mask starting point identity mismatch")
    if not all(
        Path(path).is_file() and file_sha256(Path(path)) == digest
        for path, digest in protocol.get("source_sha256", {}).items()
    ):
        raise ValueError("V4.6 source identity changed")

    rows = read_jsonl(args.train_file)
    if len(rows) != protocol["train_events"]:
        raise ValueError("V4.6 training row count mismatch")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, torch_dtype=dtype
    ).cuda()
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(base, args.mask_adapter, is_trainable=True)
    model.enable_input_require_grads()
    device = next(model.parameters()).device

    # Cache the common Mask margins before either arm updates a parameter.
    initial_margin: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for row in rows:
            prompt = str(row["prompt"])
            a = mean_completion_logprob(
                model,
                tokenizer,
                prompt,
                canonical_completion(row["action_a"]),
                args.max_length,
                device,
            )
            b = mean_completion_logprob(
                model,
                tokenizer,
                prompt,
                canonical_completion(row["action_b"]),
                args.max_length,
                device,
            )
            initial_margin[str(row["event_id"])] = float((b - a).detach())

    optimizer = torch.optim.AdamW(
        (x for x in model.parameters() if x.requires_grad), lr=args.learning_rate
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(args.output_dir / "train.jsonl")
    started, steps, accumulated = time.time(), 0, 0
    routing: Counter[str] = Counter()
    sequence: list[str] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for row in matched_epoch_order(rows, args.seed, epoch):
            event_id = str(row["event_id"])
            prompt = str(row["prompt"])
            model.train()
            logp_a = mean_completion_logprob(
                model,
                tokenizer,
                prompt,
                canonical_completion(row["action_a"]),
                args.max_length,
                device,
            )
            logp_b = mean_completion_logprob(
                model,
                tokenizer,
                prompt,
                canonical_completion(row["action_b"]),
                args.max_length,
                device,
            )
            margin = logp_b - logp_a
            m0 = torch.tensor(
                initial_margin[event_id], dtype=margin.dtype, device=device
            )
            residual = margin - m0
            anchor = residual.square()
            if args.method == "control":
                route, signed = "continue_b", residual
                primary = torch.nn.functional.softplus(
                    -args.beta * (signed - args.target_residual)
                )
            else:
                route, direction = residual_route(
                    str(row["decision"]),
                    initial_margin[event_id],
                    args.confidence_margin,
                )
                if route == "correct":
                    signed = direction * residual
                    primary = torch.nn.functional.softplus(
                        -args.beta * (signed - args.target_residual)
                    )
                else:
                    primary = args.retention_coef * anchor
            loss = primary + args.reference_anchor_coef * anchor
            (loss / args.gradient_accumulation).backward()
            accumulated += 1
            sequence.append(event_id)
            routing[route] += 1
            logger.write(
                {
                    "epoch": epoch,
                    "event_id": event_id,
                    "decision": row["decision"],
                    "route": route,
                    "initial_margin_b": initial_margin[event_id],
                    "margin_b": float(margin.detach()),
                    "residual": float(residual.detach()),
                    "loss": float(loss.detach()),
                }
            )
            if accumulated == args.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                accumulated = 0
    if accumulated:
        scale = args.gradient_accumulation / accumulated
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps += 1

    sequence_hash = hashlib.sha256("\n".join(sequence).encode()).hexdigest()
    if sequence_hash != protocol["expected_presented_event_sequence_sha256"]:
        raise RuntimeError("V4.6 event sequence differs from protocol")
    adapter = args.output_dir / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_v46_selective_residual_training",
        "method": args.method,
        "model": str(args.model),
        "base_model_sha256": directory_sha256(args.model),
        "adapter": str(adapter),
        "adapter_sha256": directory_sha256(adapter),
        "mask_adapter_sha256": directory_sha256(args.mask_adapter),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "train_file_sha256": file_sha256(args.train_file),
        "train_events": len(rows),
        **config,
        "active_event_presentations": len(sequence),
        "presented_event_sequence_sha256": sequence_hash,
        "routing_counts": dict(sorted(routing.items())),
        "optimizer_steps": steps,
        "loss_definition": "selective_signed_residual_plus_mask_margin_anchor",
        "scope": "development-only ToolSandbox learner diagnostic",
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
