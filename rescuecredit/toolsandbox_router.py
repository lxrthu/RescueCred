from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Mapping, Sequence


ROUTER_METHODS = ("margin_control", "causal_router_v5")


def desired_candidate(decision: str) -> str:
    if decision == "rescue_preference":
        return "b"
    if decision == "reverse_preference":
        return "a"
    raise ValueError(f"unsupported causal decision: {decision}")


def flip_target(mask_selected: str, decision: str) -> int:
    if mask_selected not in {"a", "b"}:
        raise ValueError(f"invalid Mask selection: {mask_selected}")
    return int(mask_selected != desired_candidate(decision))


def apply_flip(mask_selected: str, flip: bool) -> str:
    if mask_selected not in {"a", "b"}:
        raise ValueError(f"invalid Mask selection: {mask_selected}")
    if not flip:
        return mask_selected
    return "a" if mask_selected == "b" else "b"


def deterministic_group_folds(
    groups: Sequence[str], *, folds: int, seed: int
) -> list[list[int]]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    unique = sorted(set(groups))
    if len(unique) < folds:
        raise ValueError("fewer groups than requested folds")
    random.Random(seed).shuffle(unique)
    group_to_fold = {group: index % folds for index, group in enumerate(unique)}
    return [
        [index for index, group in enumerate(groups) if group_to_fold[group] == fold]
        for fold in range(folds)
    ]


def choose_flip_threshold(
    probabilities: Sequence[float],
    mask_selected: Sequence[str],
    decisions: Sequence[str],
    candidates: Sequence[float],
    *,
    min_flips: int,
) -> dict[str, Any]:
    if not (
        len(probabilities) == len(mask_selected) == len(decisions) and probabilities
    ):
        raise ValueError("threshold inputs must be aligned and non-empty")
    baseline_correct = [
        selected == desired_candidate(decision)
        for selected, decision in zip(mask_selected, decisions, strict=True)
    ]
    baseline_accuracy = sum(baseline_correct) / len(baseline_correct)
    rows = []
    for threshold in candidates:
        flips = [probability >= threshold for probability in probabilities]
        selected = [
            apply_flip(mask, flip)
            for mask, flip in zip(mask_selected, flips, strict=True)
        ]
        correct = [
            prediction == desired_candidate(decision)
            for prediction, decision in zip(selected, decisions, strict=True)
        ]
        wins = sum(
            after and not before
            for before, after in zip(baseline_correct, correct, strict=True)
        )
        losses = sum(
            before and not after
            for before, after in zip(baseline_correct, correct, strict=True)
        )
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": sum(correct) / len(correct),
                "flips": sum(flips),
                "wins": wins,
                "losses": losses,
            }
        )
    eligible = [row for row in rows if row["flips"] >= min_flips]
    if not eligible:
        selected = {
            "threshold": 1.1,
            "accuracy": baseline_accuracy,
            "flips": 0,
            "wins": 0,
            "losses": 0,
        }
    else:
        selected = max(
            eligible,
            key=lambda row: (
                row["accuracy"],
                row["wins"] - row["losses"],
                row["threshold"],
            ),
        )
    return {
        "baseline_accuracy": baseline_accuracy,
        "selected": selected,
        "candidates": rows,
    }


def completion_stats(
    model,
    tokenizer,
    prompt: str,
    completion_text: str,
    max_length: int,
    device,
):
    """Return mean completion log-probability and its frozen last-layer embedding."""

    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    action_ids = tokenizer(completion_text, add_special_tokens=False).input_ids
    if not action_ids:
        raise ValueError("router completion has no tokens")
    if len(action_ids) >= max_length:
        action_ids = action_ids[: max_length - 1]
    prompt_budget = max(1, max_length - len(action_ids))
    prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = torch.tensor([prompt_ids + action_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    base_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(base_lm, "model", None)
    if backbone is None:
        raise TypeError("causal LM does not expose its transformer backbone")
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    logits = base_lm.get_output_embeddings()(hidden)[:, :-1, :].float()
    labels = input_ids[:, 1:]
    token_logprobs = (
        torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    )
    start = len(prompt_ids) - 1
    selected_logprobs = token_logprobs[:, start : start + len(action_ids)]
    action_hidden = hidden[:, len(prompt_ids) : len(prompt_ids) + len(action_ids)]
    return (
        selected_logprobs.mean(),
        action_hidden.float().mean(dim=1).squeeze(0),
        len(action_ids),
    )


def build_router_features(
    hidden_a,
    hidden_b,
    *,
    margin_b_over_a: float,
    action_a_tokens: int,
    action_b_tokens: int,
    projection_dim: int,
    projection_seed: int,
) -> tuple[Any, Any]:
    """Build a margin-only control feature and an order-aware semantic feature."""

    import torch

    difference = (hidden_b - hidden_a).detach().float().cpu()
    difference = difference / difference.norm().clamp_min(1e-12)
    generator = torch.Generator(device="cpu").manual_seed(projection_seed)
    projection = torch.randn(
        difference.numel(), projection_dim, generator=generator, dtype=torch.float32
    ) / math.sqrt(projection_dim)
    semantic_projection = difference @ projection
    length_delta = (action_b_tokens - action_a_tokens) / max(
        1, action_a_tokens + action_b_tokens
    )
    scalars = torch.tensor(
        [margin_b_over_a, abs(margin_b_over_a), length_delta], dtype=torch.float32
    )
    return scalars, torch.cat([semantic_projection, scalars])


def standardize_features(features, mean=None, scale=None):
    features = features.float()
    if mean is None:
        mean = features.mean(dim=0)
    if scale is None:
        scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (features - mean) / scale, mean, scale


def fit_logistic_head(
    features,
    labels,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    import torch

    x, mean, scale = standardize_features(features)
    y = labels.float().reshape(-1)
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("router training requires both KEEP and FLIP labels")
    torch.manual_seed(seed)
    head = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    pos_weight = torch.tensor(negatives / positives)
    for _ in range(steps):
        logits = head(x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probabilities = torch.sigmoid(head(x).squeeze(-1))
        final_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(x).squeeze(-1), y, pos_weight=pos_weight
        )
    return {
        "weight": head.weight.detach().cpu().squeeze(0),
        "bias": head.bias.detach().cpu().squeeze(0),
        "mean": mean.detach().cpu(),
        "scale": scale.detach().cpu(),
        "train_probabilities": probabilities.detach().cpu(),
        "loss": float(final_loss),
    }


def router_probabilities(features, checkpoint: Mapping[str, Any]):
    import torch

    x, _, _ = standardize_features(
        features,
        checkpoint["mean"],
        checkpoint["scale"],
    )
    logits = x @ checkpoint["weight"] + checkpoint["bias"]
    return torch.sigmoid(logits)


def summarize_router_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("replay_valid") is True]
    decisions = Counter(str(row["decision"]) for row in valid)

    def mean(values: Sequence[float]) -> float:
        return sum(float(value) for value in values) / max(1, len(values))

    return {
        "events": len(rows),
        "valid_events": len(valid),
        "decisions": dict(sorted(decisions.items())),
        "causal_accuracy": mean([row["causal_correct"] for row in valid]),
        "rescue_accuracy": mean(
            [
                row["causal_correct"]
                for row in valid
                if row["decision"] == "rescue_preference"
            ]
        ),
        "reverse_accuracy": mean(
            [
                row["causal_correct"]
                for row in valid
                if row["decision"] == "reverse_preference"
            ]
        ),
        "flip_rate": mean([row.get("flipped", False) for row in valid]),
        "mean_selected_terminal_similarity": mean(
            [row["selected_terminal_similarity"] for row in valid]
        ),
        "mean_selected_progress_auc": mean(
            [row["selected_progress_auc"] for row in valid]
        ),
    }
