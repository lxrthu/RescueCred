from __future__ import annotations

import math
import random
import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

from rescuecredit.toolsandbox_selective_router import roc_auc


def _unique_by_event(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["event_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("paired statistics require unique event identifiers")
    return result


def align_oof_rows(
    v7_rows: Sequence[Mapping[str, Any]],
    v9_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    v7 = _unique_by_event(v7_rows)
    v9 = _unique_by_event(v9_rows)
    if set(v7) != set(v9):
        raise ValueError("V7/V9 OOF event sets differ")
    aligned = []
    for event_id in sorted(v7):
        left = v7[event_id]
        right = v9[event_id]
        if int(left["label"]) != int(right["label"]):
            raise ValueError("V7/V9 labels differ")
        if str(left["task_id_hash"]) != str(right["task_id_hash"]):
            raise ValueError("V7/V9 task identities differ")
        if bool(left["routed_to_a"]) and not bool(left["probed"]):
            raise ValueError("V7 routed an unprobed event")
        if bool(right["routed_to_a"]) and not bool(right["probed"]):
            raise ValueError("V9 routed an unprobed event")
        aligned.append(
            {
                "event_id": event_id,
                "task_id_hash": str(left["task_id_hash"]),
                "label": int(left["label"]),
                "v7_score": float(left["active_raw_score"]),
                "v9_score": float(right["active_raw_score"]),
                "v7_probed": bool(left["probed"]),
                "v9_probed": bool(right["probed"]),
                "v7_routed": bool(left["routed_to_a"]),
                "v9_routed": bool(right["routed_to_a"]),
            }
        )
    return aligned


def _operating_metrics(
    labels: Sequence[int], probed: Sequence[bool], routed: Sequence[bool]
) -> dict[str, float]:
    reverse = sum(labels)
    rescue = len(labels) - reverse
    if reverse == 0 or rescue == 0:
        raise ValueError("paired statistics require both Rescue and Reverse labels")
    return {
        "reverse_recall": sum(
            label == 1 and route
            for label, route in zip(labels, routed, strict=True)
        )
        / reverse,
        "rescue_drop": sum(
            label == 0 and route
            for label, route in zip(labels, routed, strict=True)
        )
        / rescue,
        "probe_rate": sum(probed) / len(labels),
    }


def _metrics(records: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, float]:
    labels = [int(row["label"]) for row in records]
    scores = [float(row[prefix + "_score"]) for row in records]
    operating = _operating_metrics(
        labels,
        [bool(row[prefix + "_probed"]) for row in records],
        [bool(row[prefix + "_routed"]) for row in records],
    )
    return {"roc_auc": roc_auc(labels, scores), **operating}


def _delta(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, float]:
    return {key: float(right[key]) - float(left[key]) for key in left}


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("invalid quantile request")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _interval(values: Sequence[float], alpha: float) -> dict[str, float]:
    return {
        "lower": _quantile(values, alpha / 2.0),
        "median": _quantile(values, 0.5),
        "upper": _quantile(values, 1.0 - alpha / 2.0),
        "probability_above_zero": sum(value > 0 for value in values) / len(values),
    }


def paired_task_analysis(
    v7_rows: Sequence[Mapping[str, Any]],
    v9_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    if bootstrap_replicates < 100 or permutation_replicates < 100:
        raise ValueError("paired analysis requires at least 100 replicates")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    records = align_oof_rows(v7_rows, v9_rows)
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        tasks[str(row["task_id_hash"])].append(row)
    task_ids = sorted(tasks)
    if len(task_ids) < 2:
        raise ValueError("paired analysis requires at least two tasks")

    observed_v7 = _metrics(records, "v7")
    observed_v9 = _metrics(records, "v9")
    observed_delta = _delta(observed_v7, observed_v9)
    rng = random.Random(seed)
    bootstrap: dict[str, list[float]] = {key: [] for key in observed_delta}
    attempts = 0
    maximum_attempts = bootstrap_replicates * 20
    while len(bootstrap["roc_auc"]) < bootstrap_replicates:
        attempts += 1
        if attempts > maximum_attempts:
            raise RuntimeError("too many single-class task bootstrap samples")
        sampled: list[dict[str, Any]] = []
        for task_id in rng.choices(task_ids, k=len(task_ids)):
            sampled.extend(tasks[task_id])
        labels = {int(row["label"]) for row in sampled}
        if labels != {0, 1}:
            continue
        delta = _delta(_metrics(sampled, "v7"), _metrics(sampled, "v9"))
        for key, value in delta.items():
            bootstrap[key].append(value)

    permutation: dict[str, list[float]] = {key: [] for key in observed_delta}
    for _ in range(permutation_replicates):
        permuted = []
        swaps = {task_id: bool(rng.getrandbits(1)) for task_id in task_ids}
        for row in records:
            item = dict(row)
            if swaps[str(row["task_id_hash"])]:
                for suffix in ("score", "probed", "routed"):
                    item["v7_" + suffix], item["v9_" + suffix] = (
                        item["v9_" + suffix],
                        item["v7_" + suffix],
                    )
            permuted.append(item)
        delta = _delta(_metrics(permuted, "v7"), _metrics(permuted, "v9"))
        for key, value in delta.items():
            permutation[key].append(value)

    intervals = {key: _interval(values, alpha) for key, values in bootstrap.items()}
    p_values = {}
    for key, values in permutation.items():
        observed = observed_delta[key]
        p_values[key] = {
            "two_sided": (
                1 + sum(abs(value) >= abs(observed) for value in values)
            )
            / (len(values) + 1),
            "v9_better": (1 + sum(value >= observed for value in values))
            / (len(values) + 1),
            "v9_worse": (1 + sum(value <= observed for value in values))
            / (len(values) + 1),
        }

    auc_interval = intervals["roc_auc"]
    auc_p = p_values["roc_auc"]
    if observed_delta["roc_auc"] > 0 and auc_interval["lower"] > 0 and auc_p[
        "v9_better"
    ] <= alpha:
        classification = "two_step_better"
    elif observed_delta["roc_auc"] < 0 and auc_interval["upper"] < 0 and auc_p[
        "v9_worse"
    ] <= alpha:
        classification = "two_step_worse"
    else:
        classification = "no_significant_two_step_difference"
    positive_routing_claim = (
        classification == "two_step_better"
        and observed_v9["roc_auc"] >= 0.75
        and observed_v9["reverse_recall"] >= 0.20
        and observed_v9["rescue_drop"] <= 0.02
        and observed_v9["probe_rate"] <= 0.30
    )
    task_set_hash = hashlib.sha256(
        "\n".join(task_ids).encode("utf-8")
    ).hexdigest()
    return {
        "events": len(records),
        "tasks": len(task_ids),
        "task_set_sha256": task_set_hash,
        "observed": {"v7": observed_v7, "v9": observed_v9, "v9_minus_v7": observed_delta},
        "task_bootstrap": {
            "replicates": bootstrap_replicates,
            "attempts": attempts,
            "confidence_level": 1.0 - alpha,
            "delta_intervals": intervals,
        },
        "task_swap_permutation": {
            "replicates": permutation_replicates,
            "p_values": p_values,
        },
        "classification": classification,
        "positive_routing_claim_supported": positive_routing_claim,
        "secondary_metric_p_values_exploratory": True,
        "multiplicity_correction_applied": False,
    }
