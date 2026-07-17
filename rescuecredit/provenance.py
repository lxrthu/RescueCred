from __future__ import annotations

from collections.abc import Sequence

from .types import TokenSpan


def build_token_advantages(
    num_tokens: int,
    token_spans: Sequence[TokenSpan],
    prefix_advantage: float,
    suffix_advantage: float,
) -> list[float]:
    """Route G0 to policy prefix, GH to policy suffix, and zero all non-policy tokens."""
    if num_tokens < 0:
        raise ValueError("num_tokens cannot be negative")
    output = [0.0] * num_tokens
    occupied = [False] * num_tokens
    for span in token_spans:
        if span.end > num_tokens:
            raise ValueError("token span exceeds sequence length")
        for index in range(span.start, span.end):
            if occupied[index]:
                raise ValueError("token spans must not overlap")
            occupied[index] = True
            if span.producer == "policy":
                output[index] = prefix_advantage if span.stage == "prefix" else suffix_advantage
    return output


def masked_policy_objective(log_probs: Sequence[float], advantages: Sequence[float]) -> float:
    if len(log_probs) != len(advantages):
        raise ValueError("log_probs and advantages must have equal length")
    return -sum(float(lp) * float(adv) for lp, adv in zip(log_probs, advantages))


def torch_provenance_policy_loss(log_probs, advantages, valid_mask=None):
    """Torch implementation used by the H200 trainer; import stays optional locally."""
    if valid_mask is None:
        valid_mask = advantages.ne(0)
    weights = valid_mask.to(log_probs.dtype)
    denominator = weights.sum().clamp_min(1.0)
    return -(log_probs * advantages * weights).sum() / denominator

