from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


ERROR_TERMS = {
    "error",
    "exception",
    "failed",
    "failure",
    "invalid",
    "missing",
    "not found",
    "permission denied",
    "required",
}
SUCCESS_TERMS = {"ok", "success", "succeeded", "completed", "created", "updated"}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:+-]+")
PROTECTED_RECEIPT_KEYS = {
    "branch_a",
    "branch_b",
    "causal_weight",
    "decision",
    "decision_basis",
    "ending_context_digest",
    "ground_truth",
    "label",
    "official_score",
    "official_similarity",
    "progress_auc",
    "reference_action",
    "reference_actions",
    "reward",
    "return_a",
    "return_b",
    "score_trace",
}


def _canonical_text(value: Any, *, limit: int = 4096) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)[:limit]


def _tokens(value: Any) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(_canonical_text(value))]


def _signed_hash(token: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % dimension
    sign = 1.0 if digest[8] & 1 else -1.0
    return index, sign


def _hashed_bag(values: Sequence[tuple[str, Any]], dimension: int) -> list[float]:
    result = [0.0] * dimension
    for namespace, value in values:
        for token in _tokens(value):
            index, sign = _signed_hash(namespace + ":" + token, dimension)
            result[index] += sign
    norm = math.sqrt(sum(value * value for value in result))
    if norm > 0:
        result = [value / norm for value in result]
    return result


def _contains_any(text: str, terms: set[str]) -> float:
    lowered = text.lower()
    return float(any(term in lowered for term in terms))


def _json_parseable(text: str) -> float:
    try:
        json.loads(text)
    except (TypeError, ValueError):
        return 0.0
    return 1.0


def _argument_count(action: Any) -> int:
    if not isinstance(action, Mapping):
        return 0
    arguments = action.get("arguments")
    return len(arguments) if isinstance(arguments, Mapping) else 0


def _protected_keys(value: Any) -> set[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return set()
    if isinstance(value, Mapping):
        found = {
            str(key)
            for key in value
            if str(key).casefold() in PROTECTED_RECEIPT_KEYS
        }
        for child in value.values():
            found.update(_protected_keys(child))
        return found
    if isinstance(value, (list, tuple)):
        found = set()
        for child in value:
            found.update(_protected_keys(child))
        return found
    return set()


def build_active_shadow_features(
    row: Mapping[str, Any], *, hash_dimension: int
) -> list[float]:
    """Encode only information visible after isolated one-step A/B probes.

    Full-trajectory scores, score traces, decision labels, and ending context
    digests are intentionally ignored even when the offline row contains them.
    """

    if hash_dimension < 8:
        raise ValueError("hash_dimension must be at least eight")
    branch_a = row.get("branch_a")
    branch_b = row.get("branch_b")
    if not isinstance(branch_a, Mapping) or not isinstance(branch_b, Mapping):
        raise ValueError("ActiveShadow row lacks A/B branches")
    receipts_a = branch_a.get("receipts")
    receipts_b = branch_b.get("receipts")
    if not isinstance(receipts_a, list) or not receipts_a:
        raise ValueError("ActiveShadow branch A lacks a first-step receipt")
    if not isinstance(receipts_b, list) or not receipts_b:
        raise ValueError("ActiveShadow branch B lacks a first-step receipt")
    receipt_a = receipts_a[0]
    receipt_b = receipts_b[0]
    if not isinstance(receipt_a, Mapping) or not isinstance(receipt_b, Mapping):
        raise ValueError("ActiveShadow first-step receipt is malformed")
    protected = (
        _protected_keys(receipt_a.get("content"))
        | _protected_keys(receipt_b.get("content"))
        | _protected_keys(receipt_a.get("exception"))
        | _protected_keys(receipt_b.get("exception"))
    )
    if protected:
        raise ValueError(f"ActiveShadow receipt contains protected keys: {sorted(protected)}")

    action_a = row.get("action_a")
    action_b = row.get("action_b")
    content_a = _canonical_text(receipt_a.get("content"))
    content_b = _canonical_text(receipt_b.get("content"))
    exception_a = _canonical_text(receipt_a.get("exception"))
    exception_b = _canonical_text(receipt_b.get("exception"))
    action_text_a = _canonical_text(action_a)
    action_text_b = _canonical_text(action_b)
    scalars = [
        float(bool(exception_a)),
        float(bool(exception_b)),
        float(bool(exception_a)) - float(bool(exception_b)),
        _contains_any(content_a + " " + exception_a, ERROR_TERMS),
        _contains_any(content_b + " " + exception_b, ERROR_TERMS),
        _contains_any(content_a, SUCCESS_TERMS),
        _contains_any(content_b, SUCCESS_TERMS),
        math.log1p(len(content_a)) / 10.0,
        math.log1p(len(content_b)) / 10.0,
        (len(content_b) - len(content_a)) / max(1.0, len(content_a) + len(content_b)),
        _json_parseable(content_a),
        _json_parseable(content_b),
        float(content_a == content_b),
        float(action_text_a == action_text_b),
        _argument_count(action_a) / 10.0,
        _argument_count(action_b) / 10.0,
    ]
    hashed = _hashed_bag(
        [
            ("action_a", action_a),
            ("action_b", action_b),
            ("receipt_a", content_a),
            ("receipt_b", content_b),
            ("exception_a", exception_a),
            ("exception_b", exception_b),
        ],
        hash_dimension,
    )
    return scalars + hashed


def exact_binomial_upper_bound(
    harms: int, total: int, *, alpha: float = 0.05
) -> float:
    """One-sided Clopper-Pearson upper bound solved from the binomial CDF."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if total < 0 or not 0 <= harms <= total:
        raise ValueError("invalid binomial counts")
    if total == 0 or harms == total:
        return 1.0

    def cdf(probability: float) -> float:
        return sum(
            math.comb(total, count)
            * probability**count
            * (1.0 - probability) ** (total - count)
            for count in range(harms + 1)
        )

    lower = harms / total
    upper = 1.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if cdf(middle) > alpha:
            lower = middle
        else:
            upper = middle
    return upper


def minimum_zero_harm_calibration_size(delta: float, *, alpha: float = 0.05) -> int:
    if not 0 < delta < 1 or not 0 < alpha < 1:
        raise ValueError("delta and alpha must be in (0, 1)")
    return math.ceil(math.log(alpha) / math.log(1.0 - delta))


def acquisition_mask(
    scores: Sequence[float], event_ids: Sequence[str], *, max_probe_rate: float
) -> tuple[list[bool], float]:
    if len(scores) != len(event_ids) or not scores:
        raise ValueError("acquisition inputs must be aligned and non-empty")
    if not 0 < max_probe_rate <= 1:
        raise ValueError("max_probe_rate must be in (0, 1]")
    budget = max(1, math.floor(len(scores) * max_probe_rate))
    ranked = sorted(
        range(len(scores)), key=lambda index: (-float(scores[index]), event_ids[index])
    )
    selected = set(ranked[:budget])
    mask = [index in selected for index in range(len(scores))]
    threshold = float(scores[ranked[budget - 1]])
    return mask, threshold


def active_route_metrics(
    labels: Sequence[int],
    reverse_scores: Sequence[float],
    probed: Sequence[bool],
    *,
    route_threshold: float,
    alpha: float,
) -> dict[str, Any]:
    if not (
        len(labels) == len(reverse_scores) == len(probed) and len(labels) > 0
    ):
        raise ValueError("route inputs must be aligned and non-empty")
    reverse_total = sum(int(label) for label in labels)
    rescue_total = len(labels) - reverse_total
    if reverse_total == 0 or rescue_total == 0:
        raise ValueError("route metrics require Rescue and Reverse events")
    routed_to_a = [
        bool(is_probed and float(score) >= route_threshold)
        for score, is_probed in zip(reverse_scores, probed, strict=True)
    ]
    reverse_hits = sum(
        label == 1 and route_a
        for label, route_a in zip(labels, routed_to_a, strict=True)
    )
    rescue_harms = sum(
        label == 0 and route_a
        for label, route_a in zip(labels, routed_to_a, strict=True)
    )
    return {
        "route_threshold": float(route_threshold),
        "events": len(labels),
        "probe_events": sum(probed),
        "probe_rate": sum(probed) / len(labels),
        "route_to_a": sum(routed_to_a),
        "abstain_to_b": len(labels) - sum(routed_to_a),
        "reverse_events": reverse_total,
        "rescue_events": rescue_total,
        "reverse_hits": reverse_hits,
        "rescue_harms": rescue_harms,
        "reverse_recall": reverse_hits / reverse_total,
        "rescue_drop": rescue_harms / rescue_total,
        "rescue_accuracy": 1.0 - rescue_harms / rescue_total,
        "rescue_risk_upper_bound": exact_binomial_upper_bound(
            rescue_harms, rescue_total, alpha=alpha
        ),
    }


def active_decision_metrics(
    labels: Sequence[int],
    probed: Sequence[bool],
    routed_to_a: Sequence[bool],
    *,
    alpha: float,
) -> dict[str, Any]:
    if not (len(labels) == len(probed) == len(routed_to_a) and labels):
        raise ValueError("decision inputs must be aligned and non-empty")
    if any(route_a and not is_probed for route_a, is_probed in zip(routed_to_a, probed, strict=True)):
        raise ValueError("unprobed event cannot route to A")
    reverse_total = sum(int(label) for label in labels)
    rescue_total = len(labels) - reverse_total
    if reverse_total == 0 or rescue_total == 0:
        raise ValueError("decision metrics require Rescue and Reverse events")
    reverse_hits = sum(
        label == 1 and route_a
        for label, route_a in zip(labels, routed_to_a, strict=True)
    )
    rescue_harms = sum(
        label == 0 and route_a
        for label, route_a in zip(labels, routed_to_a, strict=True)
    )
    return {
        "events": len(labels),
        "probe_events": sum(probed),
        "probe_rate": sum(probed) / len(labels),
        "route_to_a": sum(routed_to_a),
        "abstain_to_b": len(labels) - sum(routed_to_a),
        "reverse_events": reverse_total,
        "rescue_events": rescue_total,
        "reverse_hits": reverse_hits,
        "rescue_harms": rescue_harms,
        "reverse_recall": reverse_hits / reverse_total,
        "rescue_drop": rescue_harms / rescue_total,
        "rescue_accuracy": 1.0 - rescue_harms / rescue_total,
        "rescue_risk_upper_bound": exact_binomial_upper_bound(
            rescue_harms, rescue_total, alpha=alpha
        ),
    }


def choose_active_route_threshold(
    labels: Sequence[int],
    reverse_scores: Sequence[float],
    probed: Sequence[bool],
    candidates: Sequence[float],
    *,
    rescue_delta: float,
    alpha: float,
) -> dict[str, Any]:
    rows = [
        active_route_metrics(
            labels,
            reverse_scores,
            probed,
            route_threshold=threshold,
            alpha=alpha,
        )
        for threshold in candidates
    ]
    empirical = [row for row in rows if row["rescue_drop"] <= rescue_delta + 1e-12]
    if empirical:
        selected = max(
            empirical,
            key=lambda row: (
                row["reverse_recall"],
                -row["rescue_harms"],
                row["route_threshold"],
            ),
        )
    else:
        selected = active_route_metrics(
            labels,
            reverse_scores,
            probed,
            route_threshold=1.1,
            alpha=alpha,
        )
    return {
        "selected": selected,
        # Scanning thresholds and then applying a pointwise binomial bound on
        # the same labels does not preserve simultaneous coverage. Formal risk
        # certification therefore remains disabled in this feasibility helper.
        "certified_selected": None,
        "candidates": rows,
        "rescue_delta": rescue_delta,
        "alpha": alpha,
    }
