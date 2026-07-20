from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from rescuecredit.toolsandbox_active_shadow import active_decision_metrics
from rescuecredit.toolsandbox_selective_router import roc_auc


def label_from_decision(decision: str) -> int:
    if decision == "reverse_preference":
        return 1
    if decision == "rescue_preference":
        return 0
    raise ValueError(f"unsupported exact Shadow decision: {decision}")


def contract_retention(
    rows: Sequence[Mapping[str, Any]], families: Sequence[str]
) -> dict[str, Any]:
    checks = {}
    total_prevented = 0
    for family in families:
        fold = [row for row in rows if str(row["family"]) == family]
        base_harms = sum(int(row["label"]) == 0 and bool(row["base_route_to_a"]) for row in fold)
        contract_harms = sum(
            int(row["label"]) == 0 and bool(row["contract_route_to_a"]) for row in fold
        )
        base_hits = sum(int(row["label"]) == 1 and bool(row["base_route_to_a"]) for row in fold)
        contract_hits = sum(
            int(row["label"]) == 1 and bool(row["contract_route_to_a"]) for row in fold
        )
        checks[family] = {
            "base_rescue_harms": base_harms,
            "contract_rescue_harms": contract_harms,
            "base_reverse_hits": base_hits,
            "contract_reverse_hits": contract_hits,
            "no_additional_harm": contract_harms <= base_harms,
            "no_lost_reverse_hit": contract_hits >= base_hits,
        }
        total_prevented += base_harms - contract_harms
    retained = bool(
        checks
        and all(row["no_additional_harm"] and row["no_lost_reverse_hit"] for row in checks.values())
        and total_prevented >= 1
    )
    return {
        "retained": retained,
        "total_rescue_harms_prevented": total_prevented,
        "family_checks": checks,
        "rule": "no additional Rescue harm and no lost Reverse hit in every family, plus at least one aggregate Rescue harm prevented",
    }


def evaluate_deltaguard(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    baseline_scores: Mapping[str, float],
    min_class_per_family: int,
    min_auc: float,
    min_auc_gain: float,
    max_probe_rate: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    source_by_id = {str(row["event_id"]): row for row in source_rows}
    probe_by_id = {str(row["event_id"]): row for row in probe_rows}
    if len(source_by_id) != len(source_rows) or len(probe_by_id) != len(probe_rows):
        raise ValueError("DeltaGuard ledgers require unique event IDs")
    if set(probe_by_id) - set(source_by_id):
        raise ValueError("probe ledger is not a subset of source ledger")
    if set(labels) != set(source_by_id):
        raise ValueError("exact Shadow labels do not match source ledger")
    selected_ids = {
        event_id for event_id, row in source_by_id.items() if bool(row.get("selected"))
    }
    if selected_ids != set(probe_by_id):
        raise ValueError("selected source events do not match probe ledger")

    families = sorted({str(row["family"]) for row in source_rows})
    probe_eval = []
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: {"rescue": 0, "reverse": 0})
    for event_id in sorted(probe_by_id):
        row = probe_by_id[event_id]
        label = int(labels[event_id])
        family = str(source_by_id[event_id]["family"])
        coverage[family]["reverse" if label else "rescue"] += 1
        score = float(row.get("reverse_score", 0.5))
        contract_score = float(row.get("contract_reverse_score", 0.5))
        probe_eval.append(
            {
                "event_id": event_id,
                "family": family,
                "label": label,
                "score": score,
                "baseline_score": float(baseline_scores[event_id]),
                "base_route_to_a": score == 1.0,
                "contract_route_to_a": contract_score == 1.0,
            }
        )
    inconclusive = [
        f"{family}:{kind}<{min_class_per_family}"
        for family in families
        for kind in ("rescue", "reverse")
        if coverage[family][kind] < min_class_per_family
    ]
    conditional = None
    if not inconclusive:
        probe_labels = [int(row["label"]) for row in probe_eval]
        scores = [float(row["score"]) for row in probe_eval]
        controls = [float(row["baseline_score"]) for row in probe_eval]
        delta_auc = roc_auc(probe_labels, scores)
        baseline_auc = roc_auc(probe_labels, controls)
        by_family = {}
        for family in families:
            fold = [row for row in probe_eval if row["family"] == family]
            by_family[family] = {
                "events": len(fold),
                "roc_auc": roc_auc(
                    [int(row["label"]) for row in fold],
                    [float(row["score"]) for row in fold],
                ),
                "baseline_roc_auc": roc_auc(
                    [int(row["label"]) for row in fold],
                    [float(row["baseline_score"]) for row in fold],
                ),
            }
        conditional = {
            "events": len(probe_eval),
            "typed_delta_roc_auc": delta_auc,
            "v7_receipt_roc_auc": baseline_auc,
            "auc_gain_over_v7": delta_auc - baseline_auc,
            "by_held_out_family": by_family,
        }

    whole_labels = [int(labels[str(row["event_id"])]) for row in source_rows]
    probed = [bool(row.get("selected")) for row in source_rows]
    base_routes = [
        bool(probe_by_id.get(str(row["event_id"]), {}).get("reverse_score") == 1.0)
        for row in source_rows
    ]
    contract_routes = [
        bool(probe_by_id.get(str(row["event_id"]), {}).get("contract_reverse_score") == 1.0)
        for row in source_rows
    ]
    base_metrics = active_decision_metrics(
        whole_labels, probed, base_routes, alpha=alpha
    )
    contract_metrics = active_decision_metrics(
        whole_labels, probed, contract_routes, alpha=alpha
    )
    retention = contract_retention(probe_eval, families)
    passed = bool(
        not inconclusive
        and conditional is not None
        and conditional["typed_delta_roc_auc"] >= min_auc
        and conditional["auc_gain_over_v7"] >= min_auc_gain
        and base_metrics["probe_rate"] <= max_probe_rate
    )
    return {
        "status": "inconclusive" if inconclusive else "completed",
        "inconclusive_reasons": inconclusive,
        "source_events": len(source_rows),
        "probe_events": len(probe_rows),
        "families": families,
        "selected_class_coverage": dict(coverage),
        "conditional_discriminability": conditional,
        "whole_stream_public_paired_deltas": base_metrics,
        "whole_stream_contract_abstention": contract_metrics,
        "contract_retention": retention,
        "feasibility_passed": passed,
        "thresholds": {
            "min_class_per_family": min_class_per_family,
            "min_typed_delta_roc_auc": min_auc,
            "min_auc_gain_over_v7": min_auc_gain,
            "max_probe_rate": max_probe_rate,
        },
        "formal_risk_claim_made": False,
    }
