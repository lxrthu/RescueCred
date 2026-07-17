from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any

from environments.api_bank.adapter import canonical_action
from rescuecredit.estimators import PatchEMA, residual_estimate


SYSTEM_PROMPT = """You are a tool agent. Output exactly one JSON object and no prose.
Use {\"tool\": \"ToolName\", \"arguments\": {...}} for a tool call.
Use {\"type\": \"finish\"} only after all requested tool calls are complete."""

TEACHABLE_PATCHES = {"missing_required_argument", "wrong_tool_replace", "premature_finish"}


def build_prompt(task: dict[str, Any], history: list[dict[str, Any]] | None = None) -> str:
    tools = [
        {"name": tool["name"], "required": tool.get("required", []), "description": tool.get("description", "")}
        for tool in task.get("available_tools", [])
    ]
    history = history or []
    return (
        SYSTEM_PROMPT
        + "\n\nUser goal:\n"
        + str(task.get("user_goal", ""))
        + "\n\nAvailable tools:\n"
        + json.dumps(tools, ensure_ascii=False, sort_keys=True)
        + "\n\nCalls already completed:\n"
        + json.dumps(history, ensure_ascii=False, sort_keys=True)
        + "\n\nNext action:\n"
    )


def parse_action(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for position, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def group_normalize(values: list[float], epsilon: float = 1e-6) -> list[float]:
    if not values:
        return []
    center = sum(values) / len(values)
    variance = sum((value - center) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance + epsilon)
    return [(value - center) / scale for value in values]


def classify_patch(proposal: dict[str, Any] | None, expected: dict[str, Any]) -> str | None:
    if proposal is None:
        return "invalid_json"
    if proposal.get("type") == "finish":
        return "premature_finish"
    if proposal.get("tool") != expected.get("tool"):
        return "wrong_tool_replace"
    missing = set(expected.get("arguments", {})) - set(dict(proposal.get("arguments", {})))
    if missing:
        return "missing_required_argument"
    if canonical_action(proposal) != canonical_action(expected):
        return "semantic_argument_mismatch"
    return None


@dataclass
class CreditRecord:
    patch_id: str | None
    intervened: bool
    teachable: bool
    gh: float
    g0_truth: float
    assigned_prefix_score: float
    audit_probability: float | None
    audit_draw: int | None
    shadow_steps: int


class CreditAssigner:
    def __init__(self, method: str, audit_probability: float = 0.2, mu_beta: float = 0.95, seed: int = 42) -> None:
        if method not in {"naive_h_grpo", "mask_correction", "rescuecredit", "full_shadow"}:
            raise ValueError(f"unknown method: {method}")
        self.method = method
        self.audit_probability = audit_probability
        self.ema = PatchEMA(mu_beta)
        self.rng = random.Random(seed)

    def assign(self, proposal: dict[str, Any] | None, expected: dict[str, Any]) -> CreditRecord:
        patch = classify_patch(proposal, expected)
        intervened = patch is not None
        teachable = patch in TEACHABLE_PATCHES
        g0_truth = float(not intervened)
        gh = float(not intervened or teachable)
        if self.method == "naive_h_grpo":
            score, probability, draw, shadow_steps = gh, None, None, 0
        elif self.method == "mask_correction":
            score, probability, draw, shadow_steps = (gh if not intervened else 0.0), None, None, 0
        elif self.method == "full_shadow":
            score, probability, draw, shadow_steps = g0_truth, 1.0, int(intervened), int(intervened)
        elif not intervened:
            score, probability, draw, shadow_steps = gh, None, None, 0
        elif not teachable:
            score, probability, draw, shadow_steps = 0.0, None, None, 0
        elif patch in {"missing_required_argument", "premature_finish", "invalid_json"}:
            score, probability, draw, shadow_steps = 0.0, None, None, 0
        else:
            probability = self.audit_probability
            mu = self.ema.predict(patch)
            draw = int(self.rng.random() < probability)
            score = residual_estimate(mu, draw, probability, g0_truth if draw else None)
            shadow_steps = draw
            if draw:
                self.ema.update(patch, g0_truth)
        return CreditRecord(patch, intervened, teachable, gh, g0_truth, score, probability, draw, shadow_steps)
