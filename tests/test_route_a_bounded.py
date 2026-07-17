from rescuecredit.route_a_bounded import (
    EXPECTED_EVENT_FILE_SHA256,
    EXPECTED_EVENT_SET_HASH,
    bounded_gate,
    continuation_cache_key,
    summarize_bounded_results,
    trace_prefix_matches,
)


def _row(event: int, a4: float, b4: float, a8: float, b8: float) -> dict:
    return {
        "event_id": str(event),
        "mask_selected": "a",
        "v2_selected": "b",
        "horizon_prefix_match_a": True,
        "horizon_prefix_match_b": True,
        "horizons": {
            "4": {"evaluation_valid": True, "score_a": a4, "score_b": b4},
            "8": {"evaluation_valid": True, "score_a": a8, "score_b": b8},
        },
    }


def _authorize(summary: dict) -> None:
    lock = {
        "status": "frozen_before_bounded_outcomes",
        "seed": 42,
        "horizons": [4, 8],
        "events": 55,
        "event_set_hash": EXPECTED_EVENT_SET_HASH,
        "event_file_sha256": EXPECTED_EVENT_FILE_SHA256,
        "mask_results_sha256": "mask",
        "v2_results_sha256": "v2",
        "checks": {"all": True},
    }
    summary.update(
        cache_conflicts=0,
        protocol_lock_validated=True,
        protocol_lock=lock,
        seed=42,
        requested_horizons=[4, 8],
        event_set_hash=EXPECTED_EVENT_SET_HASH,
        event_file_sha256=EXPECTED_EVENT_FILE_SHA256,
        mask_results_sha256="mask",
        v2_results_sha256="v2",
        sanity_limit=None,
    )


def test_bounded_summary_keeps_horizons_separate() -> None:
    rows = [
        _row(1, 0.0, 0.0, 0.0, 1.0),
        _row(2, 1.0, 0.0, 1.0, 0.0),
    ]
    summary = summarize_bounded_results(rows, horizons=[4, 8], event_set_hash="x")
    assert summary["horizons"]["4"]["nonzero_causal_events"] == 1
    assert summary["horizons"]["8"]["nonzero_causal_events"] == 2
    assert summary["primary_horizon"] == 8
    assert summary["primary"]["v2_better_events"] == 1
    assert summary["primary"]["v2_worse_events"] == 1


def test_bounded_gate_requires_signal_and_v2_superiority() -> None:
    rows = [_row(i, 0.0, 0.0, 0.0, 1.0) for i in range(55)]
    summary = summarize_bounded_results(rows, horizons=[4, 8], event_set_hash="x")
    _authorize(summary)
    gate = bounded_gate(summary)
    assert gate["passed"] is True
    summary["cache_conflicts"] = 1
    assert bounded_gate(summary)["passed"] is False


def test_gate_fails_when_horizon_is_uninformative() -> None:
    rows = [_row(i, 0.0, 0.0, 0.5, 0.5) for i in range(55)]
    summary = summarize_bounded_results(rows, horizons=[4, 8], event_set_hash="x")
    _authorize(summary)
    assert bounded_gate(summary)["passed"] is False


def test_continuation_cache_ignores_branch_label_but_not_visible_budget() -> None:
    payload = {"instruction": "x", "history": [], "remaining_steps": 7}
    a = continuation_cache_key({**payload, "branch": "a"}, "v1")
    b = continuation_cache_key({**payload, "branch": "b"}, "v1")
    shorter = continuation_cache_key({**payload, "remaining_steps": 3}, "v1")
    assert a == b
    assert a != shorter


def test_trace_prefix_contract_accepts_horizon_and_early_stop() -> None:
    long_trace = [{"step": i} for i in range(1, 9)]
    assert trace_prefix_matches(
        {"valid": True, "termination": "horizon", "steps": 4, "trace": long_trace[:4]},
        {"valid": True, "termination": "horizon", "steps": 8, "trace": long_trace},
    )
    stopped = {"valid": True, "termination": "policy_stop", "steps": 2, "trace": long_trace[:3]}
    assert trace_prefix_matches(stopped, dict(stopped))
    assert not trace_prefix_matches(
        {"valid": True, "termination": "horizon", "steps": 4, "trace": long_trace[:4]},
        {"valid": True, "termination": "horizon", "steps": 8, "trace": [{"step": 99}] + long_trace[1:]},
    )


def test_disagreements_only_count_primary_valid_rows() -> None:
    rows = [_row(i, 0.0, 0.0, 0.0, 1.0) for i in range(3)]
    rows[0]["horizons"]["8"]["evaluation_valid"] = False
    summary = summarize_bounded_results(rows, horizons=[4, 8], event_set_hash="x")
    assert summary["selection_disagreements"] == 2


def test_gate_rejects_wrong_seed_or_horizon() -> None:
    rows = [_row(i, 0.0, 0.0, 0.0, 1.0) for i in range(55)]
    summary = summarize_bounded_results(rows, horizons=[4, 8], event_set_hash="x")
    _authorize(summary)
    summary["seed"] = 43
    assert bounded_gate(summary)["passed"] is False
