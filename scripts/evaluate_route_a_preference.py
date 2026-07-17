#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_preference import (
    completion,
    training_preference,
    validity_relation,
)
from train_route_a_preference import mean_completion_logprob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=["mask", "v2", "v3", "v31"], required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.method in {"v3", "v31"} and args.run_summary is None:
        raise ValueError("V3/V3.1 evaluation requires --run-summary")
    base_model_sha256 = directory_sha256(args.model)
    if args.run_summary is not None:
        run = json.loads(args.run_summary.read_text(encoding="utf-8"))
        if run.get("method") != args.method:
            raise ValueError("evaluation run summary has the wrong method")
        if run.get("model") != str(args.model):
            raise ValueError("evaluation model does not match the training run")
        if run.get("adapter") != str(args.adapter):
            raise ValueError("evaluation adapter path does not match the training run")
        if args.method in {"v3", "v31"}:
            if run.get("max_length") != args.max_length or run.get("fp32") != args.fp32:
                raise ValueError("V3 evaluation precision/length does not match training")
            if run.get("adapter_sha256") != directory_sha256(args.adapter):
                raise ValueError("V3 adapter does not match the training run summary")
            if run.get("base_model_sha256") != base_model_sha256:
                raise ValueError("V3 base model does not match the training run")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_jsonl(args.validation_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).cuda()
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    device = next(model.parameters()).device
    started = time.time()
    causal = []
    by_direction = defaultdict(list)
    b_margins = []
    validity_first = []
    missing_required = []
    both_valid_causal = []
    with torch.no_grad():
        for row in rows:
            prompt = str(row["prompt"])
            logp_a = mean_completion_logprob(
                model,
                tokenizer,
                prompt,
                completion(row["action_a"]),
                args.max_length,
                device,
            )
            logp_b = mean_completion_logprob(
                model,
                tokenizer,
                prompt,
                completion(row["action_b"]),
                args.max_length,
                device,
            )
            margin_b = float((logp_b - logp_a).detach())
            b_margins.append(margin_b)
            delta = float(row["delta"])
            preference = training_preference(row, "v31")
            if preference is not None:
                chosen, _, _ = preference
                target_b = chosen == row["action_b"]
                validity_correct = margin_b > 0 if target_b else margin_b < 0
                validity_first.append(validity_correct)
                relation = validity_relation(row)
                if relation == "a_invalid_b_valid":
                    missing_required.append(validity_correct)
                elif relation == "both_valid" and abs(delta) > 1e-12:
                    both_valid_causal.append(validity_correct)
            if abs(delta) <= 1e-12:
                continue
            signed_margin = margin_b if delta > 0 else -margin_b
            correct = signed_margin > 0
            causal.append((correct, signed_margin))
            by_direction["rescue" if delta > 0 else "reverse"].append(correct)
    def accuracy(values):
        return sum(bool(value) for value in values) / max(1, len(values))

    summary = {
        "status": "completed",
        "method": args.method,
        "model": str(args.model),
        "base_model_sha256": base_model_sha256,
        "adapter": str(args.adapter),
        "adapter_sha256": directory_sha256(args.adapter),
        "run_summary_sha256": (
            file_sha256(args.run_summary) if args.run_summary else None
        ),
        "validation_file_sha256": file_sha256(args.validation_file),
        "validation_events": len(rows),
        "causal_events": len(causal),
        "causal_accuracy": accuracy([item[0] for item in causal]),
        "rescue_events": len(by_direction["rescue"]),
        "rescue_accuracy": accuracy(by_direction["rescue"]),
        "reverse_events": len(by_direction["reverse"]),
        "reverse_accuracy": accuracy(by_direction["reverse"]),
        "mean_signed_causal_margin": sum(item[1] for item in causal)
        / max(1, len(causal)),
        "b_over_a_rate_all": sum(margin > 0 for margin in b_margins)
        / max(1, len(b_margins)),
        "validity_first_events": len(validity_first),
        "validity_first_accuracy": accuracy(validity_first),
        "missing_required_events": len(missing_required),
        "missing_required_accuracy": accuracy(missing_required),
        "both_valid_causal_events": len(both_valid_causal),
        "both_valid_causal_accuracy": accuracy(both_valid_causal),
        "scope": "held-out bank preference diagnostic; not AppWorld task success",
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
