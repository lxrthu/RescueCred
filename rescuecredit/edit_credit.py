from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from environments.toolsandbox.adapter import canonical_action


ABSENT = {"__editcredit_absent__": True}
NONZERO_DECISIONS = {"rescue_preference", "reverse_preference"}


@dataclass(frozen=True)
class ActionEdit:
    path: str
    value_a: Any
    value_b: Any


@dataclass(frozen=True)
class RiskControlledThreshold:
    threshold: float
    rescue_drop: float
    reverse_recall: float
    route_to_a: int
    feasible: bool


def canonical_action_edits(
    action_a: Mapping[str, Any], action_b: Mapping[str, Any]
) -> tuple[ActionEdit, ...]:
    """Return the minimal top-level tool/argument edit set from A to B.

    Values absent from one action use a typed sentinel instead of a string so
    that a real argument equal to ``"<ABSENT>"`` cannot collide with it.
    """

    a = canonical_action(action_a)
    b = canonical_action(action_b)
    edits: list[ActionEdit] = []
    if a.get("tool") != b.get("tool"):
        edits.append(ActionEdit("/tool", a.get("tool", ABSENT), b.get("tool", ABSENT)))
    arguments_a = dict(a.get("arguments", {}))
    arguments_b = dict(b.get("arguments", {}))
    for name in sorted(set(arguments_a) | set(arguments_b)):
        value_a = arguments_a[name] if name in arguments_a else ABSENT
        value_b = arguments_b[name] if name in arguments_b else ABSENT
        if value_a != value_b:
            escaped = str(name).replace("~", "~0").replace("/", "~1")
            edits.append(ActionEdit(f"/arguments/{escaped}", value_a, value_b))
    if not edits:
        raise ValueError("EditCredit requires distinct canonical actions")
    return tuple(edits)


def parse_public_preference_prompt(prompt: str) -> dict[str, Any]:
    """Recover the public payload emitted by ``public_preference_prompt``."""

    marker = "\nSelected action:"
    if marker not in prompt:
        raise ValueError("unrecognized public preference prompt")
    prefix = prompt[: prompt.rfind(marker)]
    payload_start = prefix.find("{")
    if payload_start < 0:
        raise ValueError("public preference prompt has no JSON payload")
    payload = json.loads(prefix[payload_start:])
    required = {"visible_history", "public_tool_schemas", "candidate_a", "candidate_b"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("public preference payload is incomplete")
    return payload


def edit_value_completion(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def edit_comparison_prompt(
    *,
    public_payload: Mapping[str, Any],
    edit: ActionEdit,
    swap_candidates: bool,
) -> str:
    """Build an order-randomizable, source-free prompt for one changed field."""

    candidate_a = canonical_action(public_payload["candidate_a"])
    candidate_b = canonical_action(public_payload["candidate_b"])
    left, right = (
        (candidate_b, candidate_a) if swap_candidates else (candidate_a, candidate_b)
    )
    payload = {
        "visible_history": list(public_payload["visible_history"]),
        "public_tool_schemas": list(public_payload["public_tool_schemas"]),
        "candidate_left": left,
        "candidate_right": right,
        "decision_field": edit.path,
    }
    return (
        "Choose the better value for the specified changed field using only the "
        "visible history and public tool schemas. Candidate order is arbitrary. "
        "Return exactly the canonical JSON value, with no explanation.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\nSelected field value:"
    )


def signed_direction(decision: str) -> int:
    if decision == "rescue_preference":
        return 1
    if decision == "reverse_preference":
        return -1
    raise ValueError(f"unsupported counterfactual decision: {decision}")


def credit_firewall_advantage(
    *,
    intervened: bool,
    step_index: int,
    intervention_step: int,
    assisted_advantage: float,
) -> float:
    """Block executed-B reward from proposal-A and all earlier policy tokens."""

    if intervened and step_index <= intervention_step:
        return 0.0
    return float(assisted_advantage)


def edit_credit_loss(margin_b_over_a, *, decision: str, beta: float = 1.0):
    """Pairwise loss whose sign comes only from exact paired branch outcomes."""

    import torch

    if beta <= 0:
        raise ValueError("beta must be positive")
    direction = signed_direction(decision)
    return torch.nn.functional.softplus(-float(beta) * direction * margin_b_over_a)


def intervention_policy_loss(
    proposal_logprobs,
    *,
    intervened: bool,
    step_index: int,
    intervention_step: int,
    assisted_advantage: float,
):
    """Production loss boundary preventing executed-B reward from updating A."""

    advantage = credit_firewall_advantage(
        intervened=intervened,
        step_index=step_index,
        intervention_step=intervention_step,
        assisted_advantage=assisted_advantage,
    )
    return -(proposal_logprobs.mean() * float(advantage))


def edit_credit_objective(
    policy_margin_b_over_a,
    reference_margin_b_over_a,
    *,
    decision: str,
    beta: float,
    absolute_margin_coef: float,
    target_margin: float,
    reference_anchor_coef: float,
):
    """Shared production objective used by training and gradient sanity tests."""

    import torch

    if beta <= 0 or absolute_margin_coef < 0 or target_margin < 0 or reference_anchor_coef < 0:
        raise ValueError("invalid EditCredit objective coefficient")
    direction = float(signed_direction(decision))
    policy_chosen = direction * policy_margin_b_over_a
    reference_chosen = direction * reference_margin_b_over_a
    dpo = torch.nn.functional.softplus(-float(beta) * (policy_chosen - reference_chosen))
    absolute = torch.nn.functional.softplus(
        float(beta) * (float(target_margin) - policy_chosen)
    )
    anchor = (policy_margin_b_over_a - reference_margin_b_over_a).square()
    total = dpo + float(absolute_margin_coef) * absolute + float(reference_anchor_coef) * anchor
    return total, dpo, absolute, anchor


def symmetrized_edit_margin(
    *,
    prompt: str,
    action_a: Mapping[str, Any],
    action_b: Mapping[str, Any],
    scorer: Callable[[str, str], float],
) -> float:
    """Average B-vs-A field margins under both candidate presentation orders."""

    payload = parse_public_preference_prompt(prompt)
    if canonical_action(payload["candidate_a"]) != canonical_action(action_a):
        raise ValueError("prompt candidate A differs from bound action A")
    if canonical_action(payload["candidate_b"]) != canonical_action(action_b):
        raise ValueError("prompt candidate B differs from bound action B")
    margins: list[float] = []
    for edit in canonical_action_edits(action_a, action_b):
        completion_a = edit_value_completion(edit.value_a)
        completion_b = edit_value_completion(edit.value_b)
        for swapped in (False, True):
            edit_prompt = edit_comparison_prompt(
                public_payload=payload, edit=edit, swap_candidates=swapped
            )
            margins.append(float(scorer(edit_prompt, completion_b)) - float(scorer(edit_prompt, completion_a)))
    return sum(margins) / len(margins)


def empirical_binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware empirical ROC-AUC without an sklearn dependency."""

    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and aligned")
    positives = [index for index, value in enumerate(labels) if int(value) == 1]
    negatives = [index for index, value in enumerate(labels) if int(value) == 0]
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires both classes")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if scores[positive] > scores[negative]:
                wins += 1.0
            elif scores[positive] == scores[negative]:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def select_rescue_constrained_threshold(
    rows: Iterable[Mapping[str, Any]], *, rescue_delta: float
) -> RiskControlledThreshold:
    """Maximize calibration Reverse recall under an empirical Rescue budget.

    The default is B. A is selected only when ``margin_b_over_a < threshold``.
    Candidate thresholds are fixed by calibration scores; test outcomes are
    never consulted.
    """

    if not 0.0 <= rescue_delta <= 1.0:
        raise ValueError("rescue_delta must be in [0, 1]")
    records = list(rows)
    rescue = [row for row in records if row.get("decision") == "rescue_preference"]
    reverse = [row for row in records if row.get("decision") == "reverse_preference"]
    if not rescue or not reverse:
        raise ValueError("calibration requires Rescue and Reverse events")
    scores = sorted({float(row["margin_b_over_a"]) for row in records})
    candidates = [-math.inf] + [math.nextafter(score, math.inf) for score in scores]
    feasible: list[RiskControlledThreshold] = []
    for threshold in candidates:
        rescue_harms = sum(float(row["margin_b_over_a"]) < threshold for row in rescue)
        reverse_hits = sum(float(row["margin_b_over_a"]) < threshold for row in reverse)
        rescue_drop = rescue_harms / len(rescue)
        reverse_recall = reverse_hits / len(reverse)
        candidate = RiskControlledThreshold(
            threshold=threshold,
            rescue_drop=rescue_drop,
            reverse_recall=reverse_recall,
            route_to_a=sum(float(row["margin_b_over_a"]) < threshold for row in records),
            feasible=rescue_drop <= rescue_delta + 1e-12,
        )
        if candidate.feasible:
            feasible.append(candidate)
    return max(
        feasible,
        key=lambda item: (item.reverse_recall, -item.rescue_drop, -item.route_to_a, -item.threshold),
    )


def summarize_selection(rows: Sequence[Mapping[str, Any]], *, threshold: float) -> dict[str, Any]:
    selected_rows = []
    for row in rows:
        selected = "a" if float(row["margin_b_over_a"]) < threshold else "b"
        target = "b" if row["decision"] == "rescue_preference" else "a"
        selected_rows.append({**row, "selected": selected, "target": target, "correct": selected == target})
    rescue = [row for row in selected_rows if row["decision"] == "rescue_preference"]
    reverse = [row for row in selected_rows if row["decision"] == "reverse_preference"]
    accuracy = sum(row["correct"] for row in selected_rows) / max(1, len(selected_rows))
    rescue_accuracy = sum(row["correct"] for row in rescue) / max(1, len(rescue))
    reverse_recall = sum(row["correct"] for row in reverse) / max(1, len(reverse))
    return {
        "events": len(selected_rows),
        "accuracy": accuracy,
        "rescue_accuracy": rescue_accuracy,
        "reverse_recall": reverse_recall,
        "balanced_accuracy": 0.5 * (rescue_accuracy + reverse_recall),
        "route_to_a": sum(row["selected"] == "a" for row in selected_rows),
        "rows": selected_rows,
    }


def stratified_group_folds(
    rows: Sequence[Mapping[str, Any]], *, folds: int, seed: int
) -> dict[str, int]:
    """Deterministically assign whole tasks to approximately stratified folds."""

    if folds < 3:
        raise ValueError("at least three folds are required for train/calibration/test")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        group = str(row["task_id_hash"])
        grouped.setdefault(group, []).append(row)
    if len(grouped) < folds:
        raise ValueError("fewer task groups than folds")
    rng = random.Random(seed)
    tie_breakers = {group: rng.random() for group in grouped}
    stats = []
    for group, members in grouped.items():
        rescue = sum(row.get("decision") == "rescue_preference" for row in members)
        reverse = sum(row.get("decision") == "reverse_preference" for row in members)
        stats.append((group, rescue, reverse, len(members)))
    stats.sort(key=lambda item: (-max(item[1], item[2]), -item[3], tie_breakers[item[0]], item[0]))

    totals = [{"rescue": 0, "reverse": 0, "events": 0, "groups": 0} for _ in range(folds)]
    assignment: dict[str, int] = {}
    for group, rescue, reverse, events in stats:
        fold = min(
            range(folds),
            key=lambda index: (
                totals[index]["rescue"] + rescue,
                totals[index]["reverse"] + reverse,
                totals[index]["events"] + events,
                totals[index]["groups"],
                index,
            ),
        )
        assignment[group] = fold
        totals[fold]["rescue"] += rescue
        totals[fold]["reverse"] += reverse
        totals[fold]["events"] += events
        totals[fold]["groups"] += 1
    return assignment


def fold_role(
    row: Mapping[str, Any], *, assignment: Mapping[str, int], test_fold: int, folds: int
) -> str:
    group_fold = int(assignment[str(row["task_id_hash"])])
    if group_fold == test_fold:
        return "test"
    if group_fold == (test_fold + 1) % folds:
        return "calibration"
    return "train"
