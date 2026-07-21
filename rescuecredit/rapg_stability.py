from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be in [0, 1]")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _gain(uniform_numerator: float, rapg_numerator: float) -> float:
    return (uniform_numerator - rapg_numerator) / max(uniform_numerator, 1e-12)


def summarize_task_stability(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 20_000,
    seed: int = 42,
    minimum_gain: float = 0.15,
) -> dict[str, Any]:
    """Audit whether a frozen RAPG design-MSE gain is stable across tasks.

    The audit operates only on already-computed design-variance numerators. It
    performs no model fitting, propensity tuning, or event filtering.
    """

    if len(task_rows) < 2:
        raise ValueError("task stability requires at least two tasks")
    if bootstrap_replicates < 100:
        raise ValueError("task bootstrap requires at least 100 replicates")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in task_rows:
        task_id = str(raw["task_id"])
        if task_id in seen:
            raise ValueError(f"duplicate task id: {task_id}")
        seen.add(task_id)
        events = int(raw["events"])
        uniform = float(raw["uniform_numerator"])
        rapg = float(raw["rapg_numerator"])
        if events <= 0 or not all(math.isfinite(value) for value in (uniform, rapg)):
            raise ValueError(f"invalid task row: {task_id}")
        if uniform < 0.0 or rapg < 0.0:
            raise ValueError(f"negative design numerator: {task_id}")
        improvement = uniform - rapg
        normalized.append(
            {
                "task_id": task_id,
                "events": events,
                "uniform_numerator": uniform,
                "rapg_numerator": rapg,
                "numerator_improvement": improvement,
                "relative_gain": _gain(uniform, rapg),
            }
        )

    total_uniform = math.fsum(row["uniform_numerator"] for row in normalized)
    total_rapg = math.fsum(row["rapg_numerator"] for row in normalized)
    total_events = sum(row["events"] for row in normalized)
    observed_gain = _gain(total_uniform, total_rapg)

    loo_rows: list[dict[str, Any]] = []
    for row in normalized:
        loo_uniform = total_uniform - row["uniform_numerator"]
        loo_rapg = total_rapg - row["rapg_numerator"]
        loo_rows.append(
            {
                "held_out_task": row["task_id"],
                "remaining_events": total_events - row["events"],
                "mse_gain_over_uniform": _gain(loo_uniform, loo_rapg),
            }
        )
    loo_gains = sorted(row["mse_gain_over_uniform"] for row in loo_rows)

    positive = sorted(
        (max(0.0, row["numerator_improvement"]) for row in normalized),
        reverse=True,
    )
    positive_total = math.fsum(positive)
    top1_share = positive[0] / positive_total if positive_total > 0.0 else 1.0
    top5_share = (
        math.fsum(positive[:5]) / positive_total if positive_total > 0.0 else 1.0
    )

    generator = random.Random(seed)
    task_count = len(normalized)
    bootstrap_gains: list[float] = []
    for _ in range(bootstrap_replicates):
        sampled = [normalized[generator.randrange(task_count)] for _ in range(task_count)]
        uniform = math.fsum(row["uniform_numerator"] for row in sampled)
        rapg = math.fsum(row["rapg_numerator"] for row in sampled)
        bootstrap_gains.append(_gain(uniform, rapg))
    bootstrap_gains.sort()
    bootstrap_lower = _quantile(bootstrap_gains, 0.025)
    bootstrap_median = _quantile(bootstrap_gains, 0.5)
    bootstrap_upper = _quantile(bootstrap_gains, 0.975)

    loo_minimum = min(loo_gains)
    robust = loo_minimum >= minimum_gain and bootstrap_lower > 0.0
    classification = (
        "surrogate_signal_robust_but_frozen_gate_failed"
        if robust
        else "surrogate_gain_task_concentrated"
    )
    ranked_rows = sorted(
        normalized, key=lambda row: row["numerator_improvement"], reverse=True
    )
    return {
        "classification": classification,
        "tasks": task_count,
        "events": total_events,
        "minimum_gain": minimum_gain,
        "observed_mse_gain_over_uniform": observed_gain,
        "task_improvement_fraction": sum(
            row["numerator_improvement"] > 0.0 for row in normalized
        )
        / task_count,
        "positive_improvement_concentration": {
            "top1_share": top1_share,
            "top5_share": top5_share,
        },
        "leave_one_task_out": {
            "minimum": loo_minimum,
            "median": _quantile(loo_gains, 0.5),
            "maximum": max(loo_gains),
            "all_at_least_minimum_gain": loo_minimum >= minimum_gain,
            "rows": loo_rows,
        },
        "task_cluster_bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "confidence_level": 0.95,
            "lower": bootstrap_lower,
            "median": bootstrap_median,
            "upper": bootstrap_upper,
            "probability_gain_above_zero": sum(
                value > 0.0 for value in bootstrap_gains
            )
            / bootstrap_replicates,
            "probability_gain_at_least_minimum": sum(
                value >= minimum_gain for value in bootstrap_gains
            )
            / bootstrap_replicates,
        },
        "ranked_task_rows": ranked_rows,
    }
