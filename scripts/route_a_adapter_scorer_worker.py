#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from rescuecredit.route_a_preference import completion
from train_route_a_preference import mean_completion_logprob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            prompt = str(request["prompt"])
            action_a = request["action_a"]
            action_b = request["action_b"]
            with torch.no_grad():
                logp_a = mean_completion_logprob(
                    model,
                    tokenizer,
                    prompt,
                    completion(action_a),
                    args.max_length,
                    device,
                )
                logp_b = mean_completion_logprob(
                    model,
                    tokenizer,
                    prompt,
                    completion(action_b),
                    args.max_length,
                    device,
                )
            margin = float((logp_b - logp_a).detach())
            selected = "b" if margin > 0 else "a"
            response = {
                "selected": selected,
                "action": action_b if selected == "b" else action_a,
                "b_over_a_margin": margin,
                "scoring_failed": False,
            }
        except Exception as error:
            response = {
                "selected": "a",
                "action": None,
                "scoring_failed": True,
                "error_type": type(error).__name__,
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
