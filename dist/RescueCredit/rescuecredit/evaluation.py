from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import mean


def assisted_success(records: Sequence[dict]) -> float:
    return mean(float(record["assisted_success"]) for record in records) if records else math.nan


def unassisted_success(records: Sequence[dict]) -> float:
    return mean(float(record["unassisted_success"]) for record in records) if records else math.nan


def dependence_gap(records: Sequence[dict]) -> float:
    return assisted_success(records) - unassisted_success(records)


def intervention_rate(records: Sequence[dict]) -> float:
    return mean(float(record.get("num_teachable_interventions", 0) > 0) for record in records) if records else math.nan


def first_pass_success(records: Sequence[dict]) -> float:
    return mean(float(record["first_pass_valid"]) for record in records) if records else math.nan


def mse(estimates: Iterable[float], truths: Iterable[float]) -> float:
    pairs = list(zip(estimates, truths))
    return mean((float(estimate) - float(truth)) ** 2 for estimate, truth in pairs) if pairs else math.nan


def rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    rx, ry = rankdata(xs), rankdata(ys)
    mx, my = mean(rx), mean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return numerator / denominator if denominator else math.nan

