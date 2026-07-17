#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from rescuecredit.appworld_shadow_credit import json_object
from rescuecredit.route_a_task_eval import validated_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    tokenizer.truncation_side = "left"
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
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, encoded["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            action = validated_action(json_object(text))
            response = {
                "action": action,
                "generation_failed": action is None,
                "raw_prefix": text[:500],
            }
        except Exception as error:
            response = {
                "action": None,
                "generation_failed": True,
                "error_type": type(error).__name__,
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
