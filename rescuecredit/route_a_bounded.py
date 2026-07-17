from __future__ import annotations

import hashlib
import json
from typing import Any


CONFIRMATORY_CODE_PATHS = (
    "scripts/evaluate_route_a_bounded.py",
    "scripts/freeze_route_a_bounded_confirm_protocol.py",
    "scripts/analyze_route_a_bounded_confirm.py",
    "scripts/appworld_azure_continuation_worker.py",
    "scripts/cloud/run_route_a_appworld_bounded_confirm.sh",
    "rescuecredit/route_a_bounded.py",
    "rescuecredit/appworld_shadow_credit.py",
    "rescuecredit/azure_client.py",
    "scripts/attach_appworld_shadow_credit.py",
    "scripts/audit_appworld_deployable_harness.py",
    "environments/appworld/adapter.py",
)

from rescuecredit.route_a_immediate import TOLERANCE, preferred_action


EXPECTED_SEED = 42
EXPECTED_HORIZONS = (4, 8)
EXPECTED_EVENTS = 55
EXPECTED_EVENT_SET_HASH = "395b3526ecde2ced7f5d76c1d1e280128036c73798deb8fb97b0323091690929"
EXPECTED_EVENT_FILE_SHA256 = "fcfd0de213e044c7ae54bd5a4f340b50ade39a4d1f508627adceb5f02a683f3c"
MIN_VALID_EVENTS = 40
MIN_NONZERO_EVENTS = 5


def continuation_cache_key(payload: dict[str, Any], policy_version: str) -> str:
    visible_payload = dict(payload)
    visible_payload.pop("branch", None)
    encoded = json.dumps(
        {"policy_version": policy_version, "payload": visible_payload},
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def trace_prefix_matches(short: dict[str, Any], long: dict[str, Any]) -> bool:
    if not short.get("valid") or not long.get("valid"):
        return False
    short_trace = short.get("trace", [])
    long_trace = long.get("trace", [])
    if short_trace != long_trace[: len(short_trace)]:
        return False
    termination = short.get("termination")
    if termination in {"policy_stop", "task_completed"}:
        return long.get("termination") == termination and short_trace == long_trace
    return termination == "horizon" and int(short.get("steps", 0)) == 4 and len(
        short_trace
    ) == 4


def summarize_horizon(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    key = str(int(horizon))
    valid = [
        row
        for row in rows
        if row.get("horizons", {}).get(key, {}).get("evaluation_valid")
    ]

    def scores(row: dict[str, Any]) -> tuple[float, float]:
        payload = row["horizons"][key]
        return float(payload["score_a"]), float(payload["score_b"])

    causal = [row for row in valid if abs(scores(row)[1] - scores(row)[0]) > TOLERANCE]

    def selected_score(row: dict[str, Any], method: str) -> float:
        score_a, score_b = scores(row)
        return score_b if row[f"{method}_selected"] == "b" else score_a

    def correct(row: dict[str, Any], method: str) -> bool:
        score_a, score_b = scores(row)
        return row[f"{method}_selected"] == preferred_action(score_a, score_b)

    mask_scores = [selected_score(row, "mask") for row in valid]
    v2_scores = [selected_score(row, "v2") for row in valid]
    deltas = [v2 - mask for mask, v2 in zip(mask_scores, v2_scores)]
    mask_mean = sum(mask_scores) / max(1, len(mask_scores))
    v2_mean = sum(v2_scores) / max(1, len(v2_scores))
    mask_accuracy = sum(correct(row, "mask") for row in causal) / max(1, len(causal))
    v2_accuracy = sum(correct(row, "v2") for row in causal) / max(1, len(causal))
    rescue = sum(scores(row)[1] - scores(row)[0] > TOLERANCE for row in valid)
    reverse = sum(scores(row)[1] - scores(row)[0] < -TOLERANCE for row in valid)
    return {
        "horizon": int(horizon),
        "valid_paired_events": len(valid),
        "invalid_events": len(rows) - len(valid),
        "nonzero_causal_events": len(causal),
        "rescue_preference_events": rescue,
        "reverse_preference_events": reverse,
        "zero_delta_events": len(valid) - rescue - reverse,
        "mask_mean_official_score": mask_mean,
        "v2_mean_official_score": v2_mean,
        "score_improvement": v2_mean - mask_mean,
        "mask_causal_selection_accuracy": mask_accuracy,
        "v2_causal_selection_accuracy": v2_accuracy,
        "causal_accuracy_improvement": v2_accuracy - mask_accuracy,
        "v2_better_events": sum(delta > TOLERANCE for delta in deltas),
        "v2_worse_events": sum(delta < -TOLERANCE for delta in deltas),
        "ties": sum(abs(delta) <= TOLERANCE for delta in deltas),
    }


def summarize_bounded_results(
    rows: list[dict[str, Any]], *, horizons: list[int], event_set_hash: str
) -> dict[str, Any]:
    ordered = sorted(set(int(value) for value in horizons))
    if not ordered or ordered[0] < 2:
        raise ValueError("bounded horizons must contain integers >= 2")
    summaries = {str(value): summarize_horizon(rows, value) for value in ordered}
    primary = summaries[str(max(ordered))]
    primary_key = str(max(ordered))
    primary_valid_rows = [
        row
        for row in rows
        if row.get("horizons", {}).get(primary_key, {}).get("evaluation_valid")
    ]
    disagreements = sum(
        row.get("mask_selected") != row.get("v2_selected")
        for row in primary_valid_rows
    )
    prefix_comparable = [
        row
        for row in rows
        if row.get("horizons", {}).get("4", {}).get("evaluation_valid")
        and row.get("horizons", {}).get("8", {}).get("evaluation_valid")
    ]
    prefix_mismatches = sum(
        not bool(row.get("horizon_prefix_match_a"))
        or not bool(row.get("horizon_prefix_match_b"))
        for row in prefix_comparable
    )
    prefix_unverifiable = sum(
        row.get("horizons", {}).get("8", {}).get("evaluation_valid")
        and not row.get("horizons", {}).get("4", {}).get("evaluation_valid")
        for row in rows
    )
    return {
        "status": "completed",
        "stage": "route_a_appworld_bounded_horizon_seed42",
        "event_set_hash": event_set_hash,
        "events": len(rows),
        "horizons": summaries,
        "primary_horizon": max(ordered),
        "selection_disagreements": disagreements,
        "horizon_prefix_mismatches": prefix_mismatches,
        "horizon_prefix_unverifiable": prefix_unverifiable,
        "primary": primary,
        "continuation_policy": (
            "azure_gpt4o_temperature0_visible_only_cached_v3_format_repair"
        ),
        "horizon_prefix_coupling": True,
        "continuation_input_excludes_evaluator_and_reference": True,
        "reference_suffix_used": False,
        "test_split_access": False,
        "scope": (
            "controlled-state bounded-horizon AppWorld causal diagnostic; not fully "
            "autonomous task success"
        ),
    }


def bounded_gate(summary: dict[str, Any]) -> dict[str, Any]:
    primary = summary["primary"]
    lock = summary.get("protocol_lock", {})
    checks = {
        "enough_valid_paired_events": int(primary["valid_paired_events"])
        >= MIN_VALID_EVENTS,
        "enough_nonzero_causal_events": int(primary["nonzero_causal_events"])
        >= MIN_NONZERO_EVENTS,
        "methods_make_different_selections": int(summary["selection_disagreements"])
        >= 3,
        "v2_improves_bounded_official_score": float(primary["score_improvement"])
        > TOLERANCE,
        "v2_has_more_wins_than_losses": int(primary["v2_better_events"])
        > int(primary["v2_worse_events"]),
        "v2_improves_causal_selection_accuracy": float(
            primary["causal_accuracy_improvement"]
        )
        > TOLERANCE,
        "continuation_is_reference_free": bool(
            summary.get("continuation_input_excludes_evaluator_and_reference")
        ),
        "no_reference_suffix_or_test_access": not bool(
            summary.get("reference_suffix_used")
        )
        and not bool(summary.get("test_split_access")),
        "cache_has_no_conflicts": int(summary.get("cache_conflicts", 0)) == 0,
        "h4_is_cached_prefix_of_h8_policy": bool(
            summary.get("horizon_prefix_coupling")
        ),
        "observed_horizon_prefixes_match": int(
            summary.get("horizon_prefix_mismatches", -1)
        )
        == 0,
        "all_primary_valid_prefixes_are_verifiable": int(
            summary.get("horizon_prefix_unverifiable", -1)
        )
        == 0,
        "frozen_protocol_validated": bool(summary.get("protocol_lock_validated")),
        "exact_preregistered_seed_horizons_and_event_count": int(
            summary.get("seed", -1)
        )
        == EXPECTED_SEED
        and tuple(summary.get("requested_horizons", [])) == EXPECTED_HORIZONS
        and int(summary.get("events", -1)) == EXPECTED_EVENTS
        and int(summary.get("primary_horizon", -1)) == max(EXPECTED_HORIZONS),
        "exact_frozen_event_identity": summary.get("event_set_hash")
        == EXPECTED_EVENT_SET_HASH
        and summary.get("event_file_sha256") == EXPECTED_EVENT_FILE_SHA256,
        "embedded_lock_is_consistent": lock.get("status")
        == "frozen_before_bounded_outcomes"
        and lock.get("seed") == EXPECTED_SEED
        and tuple(lock.get("horizons", [])) == EXPECTED_HORIZONS
        and lock.get("events") == EXPECTED_EVENTS
        and lock.get("event_set_hash") == EXPECTED_EVENT_SET_HASH
        and lock.get("event_file_sha256") == EXPECTED_EVENT_FILE_SHA256
        and lock.get("mask_results_sha256") == summary.get("mask_results_sha256")
        and lock.get("v2_results_sha256") == summary.get("v2_results_sha256")
        and all(lock.get("checks", {}).values()),
        "full_run_not_sanity_subset": summary.get("sanity_limit") is None,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "stage": summary["stage"],
        "primary_horizon": summary["primary_horizon"],
        "checks": checks,
        "minimum_valid_events": MIN_VALID_EVENTS,
        "minimum_nonzero_events": MIN_NONZERO_EVENTS,
        "selection_disagreements": summary["selection_disagreements"],
        **primary,
        "scope": summary["scope"],
        "next_step": (
            "freeze and run confirmatory seeds"
            if passed
            else "do not expand seeds; bounded-horizon causal advantage is not established"
        ),
    }
