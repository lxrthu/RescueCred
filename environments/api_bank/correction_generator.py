from __future__ import annotations

import hashlib
import json
from typing import Any

from .deployable import public_harness_observation


def build_visible_repair_prompt(
    observation: dict[str, Any],
    proposal: dict[str, Any],
    reason: str,
    previous_tool_result: dict[str, Any] | None = None,
) -> str:
    public = public_harness_observation(observation)
    return (
        "Repair one tool call using only the visible information below.\n"
        "Rules:\n"
        "1. Output exactly one JSON object and no prose.\n"
        "2. Keep the same tool name.\n"
        "3. Fill missing required arguments and replace an existing argument only when its current value is unsupported or contradicts visible context.\n"
        "4. Preserve every already-supported argument exactly.\n"
        "5. Never invent a value: copy exact values from the user goal or prior tool receipt.\n"
        "6. If any repair value is ambiguous or unavailable, output {}.\n\n"
        f"User goal:\n{public.get('user_goal', '')}\n\n"
        f"Visible tool schema:\n{json.dumps(public.get('available_tools', []), ensure_ascii=False, sort_keys=True)}\n\n"
        f"Prior tool receipt:\n{json.dumps(previous_tool_result, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Invalid proposal:\n{json.dumps(proposal, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Verifier feedback:\n{reason}\n\n"
        "Repaired action:\n"
    )


class FrozenModelCorrectionGenerator:
    """Deterministic frozen-model proposal generator with an in-memory cache."""

    def __init__(
        self,
        model_name_or_path: str,
        revision: str | None = None,
        device: str = "cuda",
        max_new_tokens: int = 64,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=True,
            dtype=dtype,
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.cache: dict[str, dict[str, Any] | None] = {}

    def __call__(
        self,
        observation: dict[str, Any],
        proposal: dict[str, Any],
        reason: str,
        previous_tool_result: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        import torch
        # Delayed to avoid environments.api_bank package initialization
        # cycling back into rescuecredit.training.
        from rescuecredit.training import parse_action

        prompt = build_visible_repair_prompt(observation, proposal, reason, previous_tool_result)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        if digest in self.cache:
            return self.cache[digest]
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a conservative tool-call repair engine."},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        completion = self.tokenizer.decode(
            generated[0, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        action = parse_action(completion)
        if action == {}:
            action = None
        self.cache[digest] = action
        return action
