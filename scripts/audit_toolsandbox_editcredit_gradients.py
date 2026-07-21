#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from rescuecredit.edit_credit import (
    canonical_action_edits,
    edit_comparison_prompt,
    edit_credit_objective,
    edit_value_completion,
    parse_public_preference_prompt,
)
from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import canonical_completion, event_set_hash
from scripts.freeze_toolsandbox_editcredit_protocol import STATUS
from scripts.train_route_a_preference import mean_completion_logprob


_HASH_PRIME = 2_147_483_647


def _countsketch_coefficients(seed: int) -> tuple[int, int, int, int]:
    rng = random.Random(seed)
    return (
        rng.randrange(1, _HASH_PRIME),
        rng.randrange(0, _HASH_PRIME),
        rng.randrange(1, _HASH_PRIME),
        rng.randrange(0, _HASH_PRIME),
    )


def _countsketch_maps(parameters, buckets: int, seed: int):
    import torch

    bucket_a, bucket_b, sign_a, sign_b = _countsketch_coefficients(seed)
    maps = []
    offset = 0
    for parameter in parameters:
        indices = torch.arange(
            1 + offset,
            1 + offset + parameter.numel(),
            device=parameter.device,
            dtype=torch.int64,
        )
        bucket_hash = torch.remainder(bucket_a * indices + bucket_b, _HASH_PRIME)
        sign_hash = torch.remainder(sign_a * indices + sign_b, _HASH_PRIME)
        bucket = torch.remainder(bucket_hash, buckets)
        sign = sign_hash.lt(_HASH_PRIME // 2).to(torch.float32).mul_(2.0).sub_(1.0)
        maps.append((bucket, sign))
        offset += parameter.numel()
    return maps, offset


def _trainable_parameter_sha256(named_parameters) -> str:
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _gradient_countsketch(parameters, maps, buckets: int):
    import torch

    sketch = torch.zeros(buckets, dtype=torch.float32, device=parameters[0].device)
    squared_norm = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    for parameter, (bucket, sign) in zip(parameters, maps, strict=True):
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().reshape(-1).float()
        if not torch.isfinite(gradient).all():
            raise RuntimeError("non-finite per-event gradient")
        sketch.scatter_add_(0, bucket, gradient * sign)
        squared_norm += gradient.double().square().sum()
    return sketch.detach().cpu().tolist(), float(squared_norm.sqrt().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("full_action", "editcredit"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--buckets", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != STATUS:
        raise ValueError("invalid EditCredit protocol")
    if args.method not in protocol.get("methods", []):
        raise ValueError("gradient method absent from protocol")
    if args.seed != int(protocol["config"]["seed"]):
        raise ValueError("gradient seed differs from protocol")
    if args.max_length != int(protocol["config"]["max_length"]):
        raise ValueError("gradient max length differs from protocol")
    if args.buckets != int(protocol["efficiency_config"]["gradient_sketch_buckets"]):
        raise ValueError("gradient sketch width differs from protocol")
    if file_sha256(args.train_file) != protocol.get("train_sha256"):
        raise ValueError("gradient bank differs from protocol")
    if directory_sha256(args.model) != protocol.get("base_model_sha256"):
        raise ValueError("gradient base model differs from protocol")
    source_hashes = protocol.get("source_sha256", {})
    if not source_hashes or not all(
        Path(path).is_file() and file_sha256(Path(path)) == expected
        for path, expected in source_hashes.items()
    ):
        raise ValueError("EditCredit source identity changed after freeze")

    rows = read_jsonl(args.train_file)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).cuda()
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    model = get_peft_model(
        base,
        LoraConfig(
            r=int(protocol["config"]["lora_r"]),
            lora_alpha=int(protocol["config"]["lora_alpha"]),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.enable_input_require_grads()
    # Isolate event-sampling variance: disable any model dropout while retaining
    # autograd for the trainable LoRA parameters.
    model.eval()
    device = next(model.parameters()).device
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    initial_trainable_sha256 = _trainable_parameter_sha256(named_parameters)
    parameters = [parameter for _, parameter in named_parameters]
    maps, trainable_parameters = _countsketch_maps(parameters, args.buckets, args.seed)
    started = time.time()
    records = []
    class_index: Counter[str] = Counter()
    forward_calls = 0

    for row in rows:
        decision = str(row["decision"])
        swap = bool(class_index[decision] % 2)
        class_index[decision] += 1
        prompt = str(row["prompt"])
        if args.method == "full_action":
            specs = [
                (
                    prompt,
                    canonical_completion(row["action_a"]),
                    canonical_completion(row["action_b"]),
                )
            ]
        else:
            payload = parse_public_preference_prompt(prompt)
            specs = [
                (
                    edit_comparison_prompt(
                        public_payload=payload, edit=edit, swap_candidates=swap
                    ),
                    edit_value_completion(edit.value_a),
                    edit_value_completion(edit.value_b),
                )
                for edit in canonical_action_edits(row["action_a"], row["action_b"])
            ]
        policy_margins = []
        reference_margins = []
        model.zero_grad(set_to_none=True)
        for local_prompt, completion_a, completion_b in specs:
            logp_a = mean_completion_logprob(
                model, tokenizer, local_prompt, completion_a, args.max_length, device
            )
            logp_b = mean_completion_logprob(
                model, tokenizer, local_prompt, completion_b, args.max_length, device
            )
            policy_margins.append(logp_b - logp_a)
            with torch.no_grad(), model.disable_adapter():
                ref_a = mean_completion_logprob(
                    model, tokenizer, local_prompt, completion_a, args.max_length, device
                )
                ref_b = mean_completion_logprob(
                    model, tokenizer, local_prompt, completion_b, args.max_length, device
                )
            reference_margins.append((ref_b - ref_a).detach())
            forward_calls += 4
        policy_margin = torch.stack(policy_margins).mean()
        reference_margin = torch.stack(reference_margins).mean()
        loss, *_ = edit_credit_objective(
            policy_margin,
            reference_margin,
            decision=decision,
            beta=float(protocol["config"]["beta"]),
            absolute_margin_coef=float(protocol["config"]["absolute_margin_coef"]),
            target_margin=float(protocol["config"]["target_margin"]),
            reference_anchor_coef=float(protocol["config"]["reference_anchor_coef"]),
        )
        loss.backward()
        sketch, gradient_norm = _gradient_countsketch(parameters, maps, args.buckets)
        if not math.isfinite(gradient_norm):
            raise RuntimeError("non-finite gradient norm")
        records.append(
            {
                "event_id": str(row["event_id"]),
                "task_id_hash": str(row["task_id_hash"]),
                "decision": decision,
                "method": args.method,
                "sketch": sketch,
                "gradient_norm": gradient_norm,
                "edit_fields": len(specs),
                "swap_candidates": swap if args.method == "editcredit" else None,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sketches_path = args.output_dir / "gradient_sketches.jsonl"
    write_jsonl(sketches_path, records)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_editcredit_initial_gradient_sketch",
        "method": args.method,
        "events": len(records),
        "event_set_hash": event_set_hash(records),
        "seed": args.seed,
        "buckets": args.buckets,
        "trainable_parameters": trainable_parameters,
        "initial_trainable_sha256": initial_trainable_sha256,
        "countsketch_hash": {
            "family": "independent_affine_mod_prime",
            "prime": _HASH_PRIME,
            "coefficients": list(_countsketch_coefficients(args.seed)),
        },
        "forward_calls": forward_calls,
        "mean_gradient_norm": sum(row["gradient_norm"] for row in records) / len(records),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "train_file_sha256": file_sha256(args.train_file),
        "base_model_sha256": protocol["base_model_sha256"],
        "source_sha256": source_hashes,
        "sketches_sha256": file_sha256(sketches_path),
        "wall_time_sec": time.time() - started,
        "claim_boundary": "method-specific objective gradient-noise diagnostic; not a same-estimand unbiased variance comparison",
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
