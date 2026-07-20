from __future__ import annotations

from typing import Any, Mapping, Sequence


PROBE_METHODS = ("margin_probe", "semantic_probe")


def reverse_target(decision: str) -> int:
    if decision == "reverse_preference":
        return 1
    if decision == "rescue_preference":
        return 0
    raise ValueError(f"unsupported causal decision: {decision}")


def conservative_choice(reverse_probability: float, threshold: float) -> str:
    """Default to Harness action B and select A only for confident Reverse."""

    return "a" if reverse_probability >= threshold else "b"


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires both classes")
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        raise ValueError("average precision requires a positive example")
    ordered = sorted(
        zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True
    )
    true_positives = 0
    processed = 0
    average = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        group_labels = []
        while index < len(ordered) and ordered[index][0] == score:
            group_labels.append(ordered[index][1])
            index += 1
        group_positives = sum(group_labels)
        true_positives += group_positives
        processed += len(group_labels)
        if group_positives:
            average += group_positives / positives * (true_positives / processed)
    return average


def calibration_metrics(
    labels: Sequence[int], scores: Sequence[float], *, bins: int = 10
) -> dict[str, float]:
    if len(labels) != len(scores) or not labels:
        raise ValueError("calibration inputs must be aligned and non-empty")
    brier = sum(
        (float(score) - int(label)) ** 2
        for label, score in zip(labels, scores, strict=True)
    ) / len(labels)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, score in enumerate(scores)
            if lower <= score < upper or (bin_index == bins - 1 and score == 1.0)
        ]
        if not indices:
            continue
        confidence = sum(float(scores[index]) for index in indices) / len(indices)
        frequency = sum(int(labels[index]) for index in indices) / len(indices)
        ece += len(indices) / len(labels) * abs(confidence - frequency)
    return {"brier": brier, "ece": ece}


def selective_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    if len(labels) != len(scores) or not labels:
        raise ValueError("selective inputs must be aligned and non-empty")
    reverse_count = sum(labels)
    rescue_count = len(labels) - reverse_count
    if reverse_count == 0 or rescue_count == 0:
        raise ValueError("selective metrics require Rescue and Reverse examples")
    route_to_a = [float(score) >= threshold for score in scores]
    reverse_hits = sum(
        label == 1 and selected
        for label, selected in zip(labels, route_to_a, strict=True)
    )
    rescue_harms = sum(
        label == 0 and selected
        for label, selected in zip(labels, route_to_a, strict=True)
    )
    return {
        "threshold": float(threshold),
        "events": len(labels),
        "reverse_events": reverse_count,
        "rescue_events": rescue_count,
        "route_to_a": sum(route_to_a),
        "abstain_to_b": len(labels) - sum(route_to_a),
        "reverse_hits": reverse_hits,
        "rescue_harms": rescue_harms,
        "reverse_recall": reverse_hits / reverse_count,
        "rescue_accuracy": 1.0 - rescue_harms / rescue_count,
        "rescue_drop": rescue_harms / rescue_count,
        "overall_accuracy": (reverse_hits + rescue_count - rescue_harms) / len(labels),
    }


def choose_conservative_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    candidates: Sequence[float],
    *,
    rescue_delta: float,
) -> dict[str, Any]:
    if not 0.0 <= rescue_delta < 1.0:
        raise ValueError("rescue_delta must be in [0, 1)")
    rows = [selective_metrics(labels, scores, threshold) for threshold in candidates]
    eligible = [row for row in rows if row["rescue_drop"] <= rescue_delta + 1e-12]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["reverse_recall"],
                -row["rescue_harms"],
                row["threshold"],
            ),
        )
    else:
        selected = selective_metrics(labels, scores, 1.1)
    return {
        "rescue_delta": rescue_delta,
        "selected": selected,
        "candidates": rows,
    }


def fit_platt_scaler(scores, labels, *, steps: int, learning_rate: float) -> dict[str, float]:
    import torch

    probabilities = scores.detach().float().clamp(1e-6, 1.0 - 1e-6)
    logits = torch.logit(probabilities)
    targets = labels.detach().float()
    raw_slope = torch.nn.Parameter(torch.tensor(0.5))
    intercept = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([raw_slope, intercept], lr=learning_rate)
    for _ in range(steps):
        slope = torch.nn.functional.softplus(raw_slope)
        calibrated_logits = slope * logits + intercept
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            calibrated_logits, targets
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    slope = torch.nn.functional.softplus(raw_slope)
    return {"slope": float(slope.detach()), "intercept": float(intercept.detach())}


def apply_platt_scaler(scores, calibration: Mapping[str, float]):
    import torch

    probabilities = scores.detach().float().clamp(1e-6, 1.0 - 1e-6)
    logits = torch.logit(probabilities)
    return torch.sigmoid(
        float(calibration["slope"]) * logits + float(calibration["intercept"])
    )


def fit_probe(
    features,
    labels,
    *,
    method: str,
    seed: int,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
) -> dict[str, Any]:
    import torch

    from rescuecredit.toolsandbox_router import standardize_features

    if method not in PROBE_METHODS:
        raise ValueError(f"unsupported probe method: {method}")
    x, mean, scale = standardize_features(features)
    y = labels.detach().float().reshape(-1)
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("probe training requires Rescue and Reverse labels")
    torch.manual_seed(seed)
    if method == "margin_probe":
        model = torch.nn.Linear(x.shape[1], 1)
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, 1),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    pos_weight = torch.tensor(negatives / positives)
    for _ in range(steps):
        logits = model(x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x).squeeze(-1))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x).squeeze(-1), y, pos_weight=pos_weight
        )
    return {
        "method": method,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "mean": mean.detach().cpu(),
        "scale": scale.detach().cpu(),
        "input_dim": int(x.shape[1]),
        "hidden_dim": int(hidden_dim),
        "train_probabilities": probabilities.detach().cpu(),
        "loss": float(loss),
    }


def probe_probabilities(features, checkpoint: Mapping[str, Any]):
    import torch

    from rescuecredit.toolsandbox_router import standardize_features

    method = str(checkpoint["method"])
    x, _, _ = standardize_features(features, checkpoint["mean"], checkpoint["scale"])
    if method == "margin_probe":
        model = torch.nn.Linear(int(checkpoint["input_dim"]), 1)
    elif method == "semantic_probe":
        hidden_dim = int(checkpoint["hidden_dim"])
        model = torch.nn.Sequential(
            torch.nn.Linear(int(checkpoint["input_dim"]), hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, 1),
        )
    else:
        raise ValueError(f"unsupported probe method: {method}")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x).squeeze(-1))
