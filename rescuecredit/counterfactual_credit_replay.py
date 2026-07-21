from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def _vector_metrics(vector, oracle) -> dict[str, float]:
    vector_norm = float(vector.norm())
    oracle_norm = float(oracle.norm().clamp_min(1e-12))
    error = vector - oracle
    cosine = (
        float(vector @ oracle) / (vector_norm * oracle_norm)
        if vector_norm > 0.0
        else 0.0
    )
    return {
        "norm": vector_norm,
        "cosine_with_oracle": cosine,
        "oracle_squared_distance": float(error.square().sum()),
        "oracle_relative_distance": float(error.norm()) / oracle_norm,
    }


def _stochastic_metrics(estimates, oracle, audit_counts) -> dict[str, float]:
    mean = estimates.mean(dim=0)
    oracle_norm = float(oracle.norm().clamp_min(1e-12))
    difference = estimates - oracle[None, :]
    squared_error = difference.square().sum(dim=1)
    estimate_norms = estimates.norm(dim=1).clamp_min(1e-12)
    cosines = estimates @ oracle / (estimate_norms * oracle_norm)
    return {
        "mean_norm": float(mean.norm()),
        "descriptive_monte_carlo_bias_relative": float(
            (mean - oracle).norm()
        )
        / oracle_norm,
        "projected_gradient_mse": float(squared_error.mean()),
        "projected_gradient_mse_se": float(
            squared_error.std(unbiased=True) / math.sqrt(len(estimates))
        ),
        "mean_cosine_with_oracle": float(cosines.mean()),
        "harmful_aggregate_update_rate": float(
            ((estimates @ oracle) < 0.0).float().mean()
        ),
        "mean_realized_audits": float(audit_counts.float().mean()),
        "p95_realized_audits": float(
            audit_counts.float().quantile(0.95)
        ),
    }


def task_bootstrap_gain(
    task_rows: Sequence[dict[str, Any]],
    method_error_key: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if not task_rows or replicates < 100:
        raise ValueError("task bootstrap requires rows and >=100 replicates")
    rng = random.Random(seed)
    count = len(task_rows)
    gains: list[float] = []
    for _ in range(replicates):
        sampled = [task_rows[rng.randrange(count)] for _ in range(count)]
        naive = sum(row["naive_oracle_squared_distance"] for row in sampled)
        method = sum(row[method_error_key] for row in sampled)
        gains.append((naive - method) / max(naive, 1e-12))
    gains.sort()

    def quantile(probability: float) -> float:
        position = probability * (len(gains) - 1)
        lower = int(position)
        upper = min(lower + 1, len(gains) - 1)
        weight = position - lower
        return gains[lower] * (1.0 - weight) + gains[upper] * weight

    return {
        "replicates": float(replicates),
        "lower95": quantile(0.025),
        "median": quantile(0.5),
        "upper95": quantile(0.975),
        "probability_above_zero": sum(value > 0.0 for value in gains)
        / len(gains),
    }


def replay_counterfactual_credit(
    *,
    score_sketches,
    proposal_returns,
    executed_returns,
    replaced,
    mean_predictions,
    groups: Sequence[str],
    propensities: Sequence[float] = (0.2, 0.3),
    primary_propensity: float = 0.3,
    replicates: int = 10_000,
    task_bootstrap_replicates: int = 20_000,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    score_sketches = score_sketches.float().cpu()
    proposal_returns = proposal_returns.float().cpu()
    executed_returns = executed_returns.float().cpu()
    replaced = replaced.bool().cpu()
    mean_predictions = mean_predictions.float().cpu()
    n = len(proposal_returns)
    if not (
        len(score_sketches)
        == len(executed_returns)
        == len(replaced)
        == len(mean_predictions)
        == len(groups)
        == n
    ):
        raise ValueError("counterfactual replay inputs must align")
    if score_sketches.ndim != 2 or any(
        tensor.ndim != 1
        for tensor in (proposal_returns, executed_returns, replaced, mean_predictions)
    ):
        raise ValueError("counterfactual replay tensor shapes are invalid")
    if not all(
        bool(torch.isfinite(tensor).all())
        for tensor in (
            score_sketches,
            proposal_returns,
            executed_returns,
            mean_predictions,
        )
    ):
        raise ValueError("counterfactual replay tensors must be finite")
    if n < 2 or replicates < 100:
        raise ValueError("counterfactual replay requires events and >=100 ledgers")
    frozen_propensities = tuple(float(value) for value in propensities)
    if primary_propensity not in frozen_propensities or any(
        not 0.0 < value < 1.0 for value in frozen_propensities
    ):
        raise ValueError("invalid fixed shadow propensities")
    eligible = torch.where(replaced)[0]
    if len(eligible) == 0:
        raise ValueError("counterfactual replay has no Harness replacements")

    oracle = torch.einsum("n,nd->d", proposal_returns, score_sketches) / n
    naive = torch.einsum("n,nd->d", executed_returns, score_sketches) / n
    firewall_returns = executed_returns.clone()
    firewall_returns[replaced] = 0.0
    firewall = torch.einsum("n,nd->d", firewall_returns, score_sketches) / n
    deterministic = {
        "full_shadow_oracle": _vector_metrics(oracle, oracle),
        "naive_b_to_a": _vector_metrics(naive, oracle),
        "firewall": _vector_metrics(firewall, oracle),
    }

    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_values = torch.rand(
        replicates, len(eligible), generator=generator, dtype=torch.float32
    )
    estimates_by_propensity: dict[str, Any] = {}
    stochastic: dict[str, Any] = {}
    for propensity in frozen_propensities:
        tag = f"p{int(round(propensity * 100)):02d}"
        draws = random_values < propensity
        ipw_returns = executed_returns.repeat(replicates, 1)
        ipw_returns[:, eligible] = (
            draws.float() / propensity
        ) * proposal_returns[eligible][None, :]
        aipw_returns = executed_returns.repeat(replicates, 1)
        aipw_returns[:, eligible] = mean_predictions[eligible][None, :] + (
            draws.float() / propensity
        ) * (
            proposal_returns[eligible] - mean_predictions[eligible]
        )[None, :]
        ipw_estimates = torch.einsum(
            "rn,nd->rd", ipw_returns, score_sketches
        ) / n
        aipw_estimates = torch.einsum(
            "rn,nd->rd", aipw_returns, score_sketches
        ) / n
        audit_counts = draws.sum(dim=1)
        estimates_by_propensity[tag] = {
            "draws": draws,
            "ipw": ipw_estimates,
            "aipw": aipw_estimates,
        }
        stochastic[tag] = {
            "propensity_per_replacement": propensity,
            "expected_audits": propensity * len(eligible),
            "expected_audit_rate_per_all_events": propensity * len(eligible) / n,
            "ipw": _stochastic_metrics(ipw_estimates, oracle, audit_counts),
            "aipw": _stochastic_metrics(aipw_estimates, oracle, audit_counts),
        }

    task_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        task_to_indices[str(group)].append(index)
    task_rows = []
    primary_tag = f"p{int(round(primary_propensity * 100)):02d}"
    primary_estimates = estimates_by_propensity[primary_tag]
    for task_id, indices in sorted(task_to_indices.items()):
        task_index = torch.tensor(indices, dtype=torch.long)
        task_replaced = replaced[task_index]
        if not bool(task_replaced.any()):
            continue
        task_n = len(indices)
        task_scores = score_sketches[task_index]
        task_oracle = torch.einsum(
            "n,nd->d", proposal_returns[task_index], task_scores
        ) / task_n
        task_naive = torch.einsum(
            "n,nd->d", executed_returns[task_index], task_scores
        ) / task_n
        task_firewall = torch.einsum(
            "n,nd->d", firewall_returns[task_index], task_scores
        ) / task_n
        naive_error = float((task_naive - task_oracle).square().sum())
        firewall_error = float((task_firewall - task_oracle).square().sum())
        local_ipw_returns = executed_returns[task_index].repeat(replicates, 1)
        local_aipw_returns = executed_returns[task_index].repeat(replicates, 1)
        global_eligible_positions = [
            int((eligible == index).nonzero(as_tuple=False)[0])
            for index in indices
            if bool(replaced[index])
        ]
        local_replaced_positions = torch.where(task_replaced)[0]
        local_draws = primary_estimates["draws"][:, global_eligible_positions]
        local_proposal = proposal_returns[task_index][local_replaced_positions]
        local_means = mean_predictions[task_index][local_replaced_positions]
        local_ipw_returns[:, local_replaced_positions] = (
            local_draws.float() / primary_propensity
        ) * local_proposal[None, :]
        local_aipw_returns[:, local_replaced_positions] = local_means[None, :] + (
            local_draws.float() / primary_propensity
        ) * (local_proposal - local_means)[None, :]
        task_ipw = torch.einsum(
            "rn,nd->rd", local_ipw_returns, task_scores
        ) / task_n
        task_aipw = torch.einsum(
            "rn,nd->rd", local_aipw_returns, task_scores
        ) / task_n
        ipw_mse = float((task_ipw - task_oracle[None, :]).square().sum(dim=1).mean())
        aipw_mse = float(
            (task_aipw - task_oracle[None, :]).square().sum(dim=1).mean()
        )
        task_rows.append(
            {
                "task_id": task_id,
                "events": task_n,
                "replacement_events": int(task_replaced.sum()),
                "naive_oracle_squared_distance": naive_error,
                "firewall_oracle_squared_distance": firewall_error,
                "rsc_ipw_oracle_mse": ipw_mse,
                "rsc_aipw_oracle_mse": aipw_mse,
                "firewall_improves_over_naive": firewall_error < naive_error,
                "rsc_ipw_improves_over_naive": ipw_mse < naive_error,
                "rsc_aipw_improves_over_naive": aipw_mse < naive_error,
            }
        )

    naive_distance = deterministic["naive_b_to_a"]["oracle_squared_distance"]
    firewall_distance = deterministic["firewall"]["oracle_squared_distance"]
    credit_delta = executed_returns - proposal_returns
    rescue_like = replaced & (credit_delta > 0.0)
    reverse_like = replaced & (credit_delta < 0.0)
    sketch_energy = score_sketches.square().sum(dim=1)
    naive_credit_error_energy = sketch_energy * credit_delta.square()
    firewall_credit_error_energy = sketch_energy * proposal_returns.square()
    diagnostics = {
        "replacement_events": int(replaced.sum()),
        "rescue_like_events": int(rescue_like.sum()),
        "reverse_like_events": int(reverse_like.sum()),
        "naive_rescue_like_credit_error_energy": float(
            naive_credit_error_energy[rescue_like].sum()
        ),
        "firewall_rescue_like_credit_error_energy": float(
            firewall_credit_error_energy[rescue_like].sum()
        ),
        "naive_reverse_like_credit_error_energy": float(
            naive_credit_error_energy[reverse_like].sum()
        ),
        "firewall_reverse_like_credit_error_energy": float(
            firewall_credit_error_energy[reverse_like].sum()
        ),
    }
    task_count = len(task_rows)
    firewall_improvements = [
        max(
            0.0,
            row["naive_oracle_squared_distance"]
            - row["firewall_oracle_squared_distance"],
        )
        for row in task_rows
    ]
    aipw_improvements = [
        max(
            0.0,
            row["naive_oracle_squared_distance"]
            - row["rsc_aipw_oracle_mse"],
        )
        for row in task_rows
    ]
    firewall_positive_total = sum(firewall_improvements)
    aipw_positive_total = sum(aipw_improvements)
    task_bootstrap = {
        "firewall_vs_naive": task_bootstrap_gain(
            task_rows,
            "firewall_oracle_squared_distance",
            replicates=task_bootstrap_replicates,
            seed=seed + 4101,
        ),
        "rsc_aipw_vs_naive": task_bootstrap_gain(
            task_rows,
            "rsc_aipw_oracle_mse",
            replicates=task_bootstrap_replicates,
            seed=seed + 4102,
        ),
    }
    analytic_expectation = executed_returns.clone()
    analytic_expectation[replaced] = proposal_returns[replaced]
    analytic_gradient = torch.einsum(
        "n,nd->d", analytic_expectation, score_sketches
    ) / n
    summary = {
        "events": n,
        "tasks": len(task_to_indices),
        "tasks_with_replacements": task_count,
        "replicates": replicates,
        "task_bootstrap_replicates": task_bootstrap_replicates,
        "seed": seed,
        "primary_propensity": primary_propensity,
        "analytic_aipw_identity": bool(
            torch.allclose(analytic_gradient, oracle, atol=1e-7, rtol=1e-7)
        ),
        "deterministic_methods": deterministic,
        "randomized_shadow": stochastic,
        "diagnostics": diagnostics,
        "firewall_oracle_distance_ratio_vs_naive": firewall_distance
        / max(naive_distance, 1e-12),
        "task_results": {
            "firewall_task_improvement_fraction_vs_naive": sum(
                row["firewall_improves_over_naive"] for row in task_rows
            )
            / max(task_count, 1),
            "rsc_ipw_task_improvement_fraction_vs_naive": sum(
                row["rsc_ipw_improves_over_naive"] for row in task_rows
            )
            / max(task_count, 1),
            "rsc_aipw_task_improvement_fraction_vs_naive": sum(
                row["rsc_aipw_improves_over_naive"] for row in task_rows
            )
            / max(task_count, 1),
            "firewall_positive_improvement_top1_share": max(
                firewall_improvements, default=0.0
            )
            / max(firewall_positive_total, 1e-12),
            "rsc_aipw_positive_improvement_top1_share": max(
                aipw_improvements, default=0.0
            )
            / max(aipw_positive_total, 1e-12),
            "bootstrap": task_bootstrap,
            "rows": task_rows,
        },
    }
    artifact = {
        "oracle_gradient": oracle,
        "naive_gradient": naive,
        "firewall_gradient": firewall,
    }
    return summary, artifact
