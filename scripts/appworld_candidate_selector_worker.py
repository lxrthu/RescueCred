#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from typing import Any


def _parse_index(text: str, size: int) -> int | None:
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    index = int(match.group())
    return index if 0 <= index < size else None


def _mean_label_logprob(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    labels: list[str],
    device: str,
    batch_size: int = 32,
) -> list[float]:
    import torch

    if len(prompts) != len(labels):
        raise ValueError("prompt/label length mismatch")
    encoded: list[tuple[list[int], int, int]] = []
    for prompt, label in zip(prompts, labels):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        label_ids = tokenizer.encode(label, add_special_tokens=False)
        if not prompt_ids or not label_ids:
            raise ValueError("empty pointwise sequence")
        encoded.append((prompt_ids + label_ids, len(prompt_ids), len(label_ids)))

    scores: list[float] = []
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        width = max(len(ids) for ids, _, _ in chunk)
        input_ids = torch.full(
            (len(chunk), width),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, (ids, _, _) in enumerate(chunk):
            input_ids[row, : len(ids)] = torch.tensor(ids, device=device)
            attention_mask[row, : len(ids)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        for row, (ids, prompt_length, label_length) in enumerate(chunk):
            prediction_positions = [
                prompt_length + offset - 1 for offset in range(label_length)
            ]
            target_tokens = torch.tensor(
                [ids[prompt_length + offset] for offset in range(label_length)],
                device=device,
            )
            selected_logits = logits[row, prediction_positions, :].float()
            selected_log_probs = torch.log_softmax(selected_logits, dim=-1)
            token_scores = selected_log_probs.gather(
                1, target_tokens.unsqueeze(1)
            ).squeeze(1)
            scores.append(float(token_scores.mean().item()))
        del logits, input_ids, attention_mask
    return scores


def _pointwise_select(
    request: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: str,
    min_probability: float,
    min_gap: float,
) -> dict[str, Any]:
    candidates = list(request.get("candidates", []))
    sources = list(request.get("candidate_sources", []))
    origins = list(request.get("candidate_origins", []))
    public_schema = request.get("public_schema", {})
    prompts: list[str] = []
    for index, candidate in enumerate(candidates):
        evidence = sources[index] if index < len(sources) else []
        provenance = origins[index] if index < len(origins) else []
        user_prompt = (
            "Judge one candidate independently. Decide whether it is the exact visible value "
            "required by the missing tool argument. Use only the task, public OpenAPI schema, "
            "argument name, candidate value, and evidence label. Do not assume that one of the "
            "candidates must be correct. Answer only Yes or No.\n\n"
            f"Task: {request.get('instruction', '')}\n"
            f"Tool: {request.get('tool', '')}\n"
            f"Missing argument: {request.get('parameter', '')}\n"
            f"Public OpenAPI schema: {json.dumps(public_schema, ensure_ascii=False, sort_keys=True)}\n"
            f"Candidate: {json.dumps(candidate, ensure_ascii=False, sort_keys=True)}\n"
            f"Candidate evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
            f"Candidate provenance: {json.dumps(provenance, ensure_ascii=False)}\n"
            "Answer:"
        )
        if getattr(tokenizer, "chat_template", None):
            user_prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a conservative reference-free tool-argument validator.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(user_prompt)

    paired_prompts: list[str] = []
    labels: list[str] = []
    for prompt in prompts:
        paired_prompts.extend([prompt, prompt])
        labels.extend(["Yes", "No"])
    label_scores = _mean_label_logprob(
        model,
        tokenizer,
        paired_prompts,
        labels,
        device,
    )
    probabilities: list[float] = []
    for index in range(len(candidates)):
        yes_score = label_scores[2 * index]
        no_score = label_scores[2 * index + 1]
        difference = no_score - yes_score
        if difference >= 0:
            exp_negative = math.exp(-difference)
            probability = exp_negative / (1.0 + exp_negative)
        else:
            probability = 1.0 / (1.0 + math.exp(difference))
        probabilities.append(probability)
    order = sorted(range(len(candidates)), key=probabilities.__getitem__, reverse=True)
    if not order:
        return {"index": None, "strategy": "pointwise_yes_no"}
    best = order[0]
    runner_up = probabilities[order[1]] if len(order) > 1 else 0.0
    gap = probabilities[best] - runner_up
    selected = best if probabilities[best] >= min_probability and gap >= min_gap else None
    return {
        "index": selected,
        "strategy": "pointwise_yes_no",
        "top_probability": probabilities[best],
        "probability_gap": gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--min-pointwise-probability", type=float, default=0.80)
    parser.add_argument("--min-pointwise-gap", type=float, default=0.20)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()

    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            candidates = list(request.get("candidates", []))
            if not candidates:
                response: dict[str, Any] = {"index": None}
            else:
                response = _pointwise_select(
                    request,
                    model,
                    tokenizer,
                    args.device,
                    args.min_pointwise_probability,
                    args.min_pointwise_gap,
                )
        except Exception as error:
            response = {"index": None, "error_type": type(error).__name__}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
