from __future__ import annotations

import math

from .types import RescueEvent


def eligible_correction(event: RescueEvent, threshold: float = 0.95) -> bool:
    correction_confidence = float(event.metadata.get("correction_verifier_confidence", 0.0))
    correction_valid = bool(event.metadata.get("correction_verified", False))
    return bool(
        event.teachable_patch
        and not event.permanent_safety_patch
        and event.correction_text
        and event.correction_text != event.proposal_text
        and correction_valid
        and correction_confidence >= threshold
    )


def pairwise_logistic_loss(chosen_logprob: float, rejected_logprob: float) -> float:
    delta = chosen_logprob - rejected_logprob
    return math.log1p(math.exp(-abs(delta))) + max(-delta, 0.0)


def torch_pairwise_loss(chosen_logprobs, rejected_logprobs):
    import torch.nn.functional as functional

    return -functional.logsigmoid(chosen_logprobs - rejected_logprobs).mean()
