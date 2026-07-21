from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any, Sequence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:+-]+")


def stable_seed(seed: int, identity: str) -> int:
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def public_hash_features(
    values: Sequence[Any], *, dimension: int = 128
) -> list[float]:
    """Signed hashing features over explicitly public values.

    Callers choose the values. This function never traverses an event row and
    therefore cannot accidentally absorb private outcome fields.
    """

    if dimension < 8:
        raise ValueError("feature dimension must be at least eight")
    text = json.dumps(values, ensure_ascii=False, sort_keys=True, default=str)
    output = [0.0] * dimension
    for token in TOKEN_PATTERN.findall(text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        output[bucket] += sign
    norm = math.sqrt(sum(value * value for value in output))
    if norm > 0:
        output = [value / norm for value in output]
    return output


def solve_expected_budget(
    weights: Sequence[float],
    *,
    expected_budget: float,
    p_min: float,
    costs: Sequence[float] | None = None,
) -> list[float]:
    """Solve p_i=clip(lambda*w_i/sqrt(c_i), p_min, 1).

    The budget is an expected cost, not a realized hard cap.
    """

    if not weights:
        return []
    if not 0.0 < p_min <= 1.0:
        raise ValueError("p_min must be in (0, 1]")
    if costs is None:
        costs = [1.0] * len(weights)
    if len(costs) != len(weights):
        raise ValueError("weights and costs must align")
    clean_weights = [max(float(value), 1e-12) for value in weights]
    clean_costs = [float(value) for value in costs]
    if any(not math.isfinite(value) or value <= 0 for value in clean_costs):
        raise ValueError("costs must be finite and positive")
    minimum = p_min * sum(clean_costs)
    maximum = sum(clean_costs)
    if expected_budget < minimum - 1e-9:
        raise ValueError(
            f"expected budget {expected_budget} is below positivity floor {minimum}"
        )
    if expected_budget >= maximum - 1e-12:
        return [1.0] * len(weights)

    def probabilities(scale: float) -> list[float]:
        return [
            min(1.0, max(p_min, scale * weight / math.sqrt(cost)))
            for weight, cost in zip(clean_weights, clean_costs, strict=True)
        ]

    low, high = 0.0, 1.0
    while sum(
        probability * cost
        for probability, cost in zip(
            probabilities(high), clean_costs, strict=True
        )
    ) < expected_budget:
        high *= 2.0
    for _ in range(100):
        middle = (low + high) / 2.0
        spent = sum(
            probability * cost
            for probability, cost in zip(
                probabilities(middle), clean_costs, strict=True
            )
        )
        if spent < expected_budget:
            low = middle
        else:
            high = middle
    result = probabilities(high)
    if abs(
        sum(p * c for p, c in zip(result, clean_costs, strict=True))
        - expected_budget
    ) > 1e-6:
        raise RuntimeError("failed to solve expected audit budget")
    return result


def _ridge_predict(train_x, train_y, evaluation_x, *, alpha: float):
    import torch

    x = train_x.double()
    y = train_y.double().reshape(-1, 1)
    z = evaluation_x.double()
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / scale
    z = (z - mean) / scale
    x = torch.cat([torch.ones(len(x), 1, dtype=x.dtype), x], dim=1)
    z = torch.cat([torch.ones(len(z), 1, dtype=z.dtype), z], dim=1)
    penalty = torch.eye(x.shape[1], dtype=x.dtype) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = torch.linalg.solve(x.T @ x + penalty, x.T @ y)
    return (z @ coefficients).reshape(-1).float()


def cross_fitted_residual_predictions(
    features,
    outcomes,
    groups: Sequence[str],
    *,
    folds: int,
    seed: int,
    ridge_alpha: float,
):
    """Task-cross-fitted mean and residual-scale predictions."""

    import torch

    from rescuecredit.toolsandbox_router import deterministic_group_folds

    if len(features) != len(outcomes) or len(groups) != len(outcomes):
        raise ValueError("cross-fit inputs must align")
    unique_groups = set(groups)
    if len(unique_groups) < folds:
        raise ValueError("fewer task groups than requested outer folds")
    outer = deterministic_group_folds(groups, folds=folds, seed=seed)
    all_indices = set(range(len(groups)))
    means = torch.zeros(len(groups), dtype=torch.float32)
    scales = torch.zeros(len(groups), dtype=torch.float32)
    fold_ids = [-1] * len(groups)
    audits: list[dict[str, Any]] = []
    for fold_id, evaluation_indices in enumerate(outer):
        training_indices = sorted(all_indices - set(evaluation_indices))
        training_groups = [groups[index] for index in training_indices]
        evaluation_groups = {groups[index] for index in evaluation_indices}
        if set(training_groups) & evaluation_groups:
            raise RuntimeError("task leakage in outer RAPG cross-fit")
        means[evaluation_indices] = _ridge_predict(
            features[training_indices],
            outcomes[training_indices],
            features[evaluation_indices],
            alpha=ridge_alpha,
        )

        inner_fold_count = min(max(2, folds - 1), len(set(training_groups)))
        inner = deterministic_group_folds(
            training_groups,
            folds=inner_fold_count,
            seed=seed + 1009 * (fold_id + 1),
        )
        local_all = set(range(len(training_indices)))
        inner_means = torch.zeros(len(training_indices), dtype=torch.float32)
        for inner_eval_local in inner:
            inner_train_local = sorted(local_all - set(inner_eval_local))
            global_train = [training_indices[index] for index in inner_train_local]
            global_eval = [training_indices[index] for index in inner_eval_local]
            inner_means[inner_eval_local] = _ridge_predict(
                features[global_train],
                outcomes[global_train],
                features[global_eval],
                alpha=ridge_alpha,
            )
        residual_targets = (
            outcomes[training_indices].float() - inner_means
        ).square().clamp_min(1e-8).log()
        log_variance = _ridge_predict(
            features[training_indices],
            residual_targets,
            features[evaluation_indices],
            alpha=ridge_alpha,
        )
        scales[evaluation_indices] = (0.5 * log_variance).exp().clamp(
            min=1e-4, max=10.0
        )
        for index in evaluation_indices:
            fold_ids[index] = fold_id
        audits.append(
            {
                "fold": fold_id,
                "training_events": len(training_indices),
                "evaluation_events": len(evaluation_indices),
                "training_tasks": len(set(training_groups)),
                "evaluation_tasks": len(evaluation_groups),
                "task_overlap": 0,
            }
        )
    return means, scales, fold_ids, audits


def build_fixed_propensities(
    *,
    score_norms,
    replaced,
    scale_predictions,
    audit_rate: float,
    p_min: float,
    outcomes=None,
    mean_predictions=None,
) -> dict[str, Any]:
    """Build fixed matched-cost propensities before audit resampling."""

    import torch

    score_norms = score_norms.float().cpu()
    replaced = replaced.bool().cpu()
    scale_predictions = scale_predictions.float().cpu()
    n = len(score_norms)
    eligible = torch.where(replaced)[0].tolist()
    expected_budget = float(n) * float(audit_rate)
    if not eligible or expected_budget > len(eligible) + 1e-9:
        raise ValueError("invalid replacement coverage for requested audit budget")
    method_weights = {
        "uniform": torch.ones(n),
        "residual_only": scale_predictions,
        "score_only": score_norms,
        "rapg": score_norms * scale_predictions,
    }
    if outcomes is not None or mean_predictions is not None:
        if outcomes is None or mean_predictions is None:
            raise ValueError("oracle allocation requires outcomes and mean predictions")
        residual = (outcomes.float().cpu() - mean_predictions.float().cpu()).abs()
        method_weights["oracle"] = score_norms * residual.clamp_min(1e-8)
    probabilities: dict[str, Any] = {}
    for method, weights in method_weights.items():
        selected = solve_expected_budget(
            [float(weights[index]) for index in eligible],
            expected_budget=expected_budget,
            p_min=p_min,
        )
        values = torch.zeros(n, dtype=torch.float32)
        values[eligible] = torch.tensor(selected, dtype=torch.float32)
        probabilities[method] = values
    return probabilities


def simulate_fixed_propensity_audits(
    *,
    score_sketches,
    score_norms,
    outcomes,
    executed_returns,
    replaced,
    mean_predictions,
    scale_predictions,
    groups: Sequence[str],
    audit_rate: float,
    p_min: float,
    replicates: int,
    seed: int,
    probabilities_by_method=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run matched-cost RAPG audit resampling on a sealed proposal batch."""

    import torch

    score_sketches = score_sketches.float().cpu()
    score_norms = score_norms.float().cpu()
    outcomes = outcomes.float().cpu()
    executed_returns = executed_returns.float().cpu()
    replaced = replaced.bool().cpu()
    mean_predictions = mean_predictions.float().cpu()
    scale_predictions = scale_predictions.float().cpu()
    n = len(outcomes)
    if not (
        len(score_sketches)
        == len(score_norms)
        == len(executed_returns)
        == len(replaced)
        == len(mean_predictions)
        == len(scale_predictions)
        == len(groups)
        == n
    ):
        raise ValueError("simulation inputs must align")
    if not 0 < audit_rate < 1 or replicates < 100:
        raise ValueError("invalid audit rate or too few replicates")
    eligible = torch.where(replaced)[0].tolist()
    expected_budget = float(n) * float(audit_rate)
    if not eligible:
        raise ValueError("RAPG batch has no path-changing replacements")
    if expected_budget > len(eligible) + 1e-9:
        raise ValueError("audit budget exceeds replacement-eligible events")
    if probabilities_by_method is None:
        probabilities_by_method = build_fixed_propensities(
            score_norms=score_norms,
            replaced=replaced,
            scale_predictions=scale_predictions,
            audit_rate=audit_rate,
            p_min=p_min,
            outcomes=outcomes,
            mean_predictions=mean_predictions,
        )
    required_methods = {
        "uniform", "residual_only", "score_only", "rapg", "oracle"
    }
    if set(probabilities_by_method) != required_methods:
        raise ValueError("fixed propensity methods are incomplete")
    full_gradient = (score_sketches * outcomes[:, None]).mean(dim=0)
    full_norm = float(full_gradient.norm().clamp_min(1e-12))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    estimates_by_method: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    task_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        task_to_indices[str(group)].append(index)

    for method, raw_probabilities in probabilities_by_method.items():
        probabilities = raw_probabilities.float().cpu()
        if len(probabilities) != n or bool((probabilities[~replaced] != 0).any()):
            raise ValueError("fixed propensity vector is invalid")
        selected_probabilities = [float(probabilities[index]) for index in eligible]
        if abs(sum(selected_probabilities) - expected_budget) > 1e-5:
            raise ValueError("fixed propensities violate the matched expected budget")
        random_values = torch.rand(
            replicates, len(eligible), generator=generator, dtype=torch.float32
        )
        draws = random_values < probabilities[eligible][None, :]
        estimated_returns = executed_returns.repeat(replicates, 1)
        estimated_returns[:, eligible] = mean_predictions[eligible][None, :] + (
            draws.float() / probabilities[eligible][None, :]
        ) * (outcomes[eligible] - mean_predictions[eligible])[None, :]
        estimates = torch.einsum(
            "rn,nd->rd", estimated_returns, score_sketches
        ) / float(n)
        estimates_by_method[method] = {
            "probabilities": probabilities,
            "audit_draws": draws,
            "gradient_estimates": estimates,
        }
        difference = estimates - full_gradient[None, :]
        squared_error = difference.square().sum(dim=1)
        mean_estimate = estimates.mean(dim=0)
        bias = mean_estimate - full_gradient
        bias_relative = float(bias.norm() / max(full_norm, 1e-12))
        standard_error_norm = float(
            estimates.var(dim=0, unbiased=True).sum().sqrt()
            / math.sqrt(replicates)
        )
        bias_upper95 = (
            float(bias.norm()) + 1.96 * standard_error_norm
        ) / max(full_norm, 1e-12)
        estimate_norms = estimates.norm(dim=1).clamp_min(1e-12)
        cosine = (
            estimates @ full_gradient
            / (estimate_norms * max(full_norm, 1e-12))
        )
        realized_cost = draws.sum(dim=1).float()
        task_mse: dict[str, float] = {}
        for group, indices in task_to_indices.items():
            group_eligible = [index for index in indices if bool(replaced[index])]
            task_mse[group] = sum(
                float(score_norms[index].square())
                * float((outcomes[index] - mean_predictions[index]).square())
                * (1.0 - float(probabilities[index]))
                / float(probabilities[index])
                for index in group_eligible
            ) / float(len(indices) ** 2)
        design_mse = sum(
            float(score_norms[index].square())
            * float((outcomes[index] - mean_predictions[index]).square())
            * (1.0 - float(probabilities[index]))
            / float(probabilities[index])
            for index in eligible
        ) / float(n**2)
        expected_ess = len(eligible) ** 2 / sum(
            1.0 / probability for probability in selected_probabilities
        )
        summaries[method] = {
            "projected_bias_relative": bias_relative,
            "projected_bias_upper95": bias_upper95,
            "projected_gradient_mse": float(squared_error.mean()),
            "projected_gradient_mse_se": float(
                squared_error.std(unbiased=True) / math.sqrt(replicates)
            ),
            "mean_cosine_similarity": float(cosine.mean()),
            "full_gradient_design_mse": design_mse,
            "expected_audits": float(probabilities.sum()),
            "mean_realized_audits": float(realized_cost.mean()),
            "realized_audits_q95": float(torch.quantile(realized_cost, 0.95)),
            "realized_audits_q99": float(torch.quantile(realized_cost, 0.99)),
            "expected_ess": float(expected_ess),
            "max_inverse_propensity": max(
                1.0 / probability for probability in selected_probabilities
            ),
            "minimum_propensity": min(selected_probabilities),
            "maximum_propensity": max(selected_probabilities),
            "task_mse": task_mse,
        }
    uniform_mse = summaries["uniform"]["full_gradient_design_mse"]
    rapg_mse = summaries["rapg"]["full_gradient_design_mse"]
    residual_mse = summaries["residual_only"]["full_gradient_design_mse"]
    task_improvements = {
        group: summaries["uniform"]["task_mse"][group]
        - summaries["rapg"]["task_mse"][group]
        for group in task_to_indices
    }
    positive_improvements = [max(0.0, value) for value in task_improvements.values()]
    total_positive = sum(positive_improvements)
    summary = {
        "events": n,
        "tasks": len(task_to_indices),
        "replacement_events": len(eligible),
        "replicates": replicates,
        "audit_rate_per_all_events": audit_rate,
        "expected_audit_budget": expected_budget,
        "p_min": p_min,
        "projected_gradient_dimension": int(score_sketches.shape[1]),
        "full_projected_gradient_norm": full_norm,
        "methods": summaries,
        "rapg_mse_gain_over_uniform": (
            uniform_mse - rapg_mse
        ) / max(uniform_mse, 1e-12),
        "rapg_mse_gain_over_residual_only": (
            residual_mse - rapg_mse
        ) / max(residual_mse, 1e-12),
        "task_improvement_fraction": sum(
            value > 0 for value in task_improvements.values()
        )
        / len(task_improvements),
        "max_positive_task_improvement_share": (
            max(positive_improvements, default=0.0) / total_positive
            if total_positive > 0
            else 1.0
        ),
        "task_improvements": task_improvements,
        "primary_weights_clipped": False,
    }
    artifact = {
        "full_gradient": full_gradient,
        "methods": estimates_by_method,
    }
    return summary, artifact
