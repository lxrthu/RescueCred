from rescuecredit.route_a_immediate import (
    causal_decision,
    immediate_gate,
    summarize_immediate_results,
)


def _row(event: int, a: float, b: float, mask: str, v2: str) -> dict:
    return {
        "event_id": str(event),
        "evaluation_valid": True,
        "score_a": a,
        "score_b": b,
        "mask_selected": mask,
        "v2_selected": v2,
    }


def test_causal_decision_is_three_way() -> None:
    assert causal_decision(0.2, 0.7) == "rescue_preference"
    assert causal_decision(0.7, 0.2) == "reverse_preference"
    assert causal_decision(0.4, 0.4) == "zero_delta"


def test_summary_scores_each_methods_selected_action() -> None:
    rows = [
        _row(1, 0.0, 1.0, "a", "b"),
        _row(2, 1.0, 0.0, "b", "a"),
        _row(3, 0.5, 0.5, "b", "b"),
    ]
    summary = summarize_immediate_results(rows, event_set_hash="hash")
    assert summary["nonzero_immediate_events"] == 2
    assert summary["selection_disagreements"] == 2
    assert summary["mask_causal_selection_accuracy"] == 0.0
    assert summary["v2_causal_selection_accuracy"] == 1.0
    assert summary["v2_better_events"] == 2
    assert summary["v2_worse_events"] == 0
    assert summary["continuation_used"] is False


def test_gate_is_preregistered_and_strict() -> None:
    rows = []
    for event in range(6):
        if event < 5:
            rows.append(_row(event, 0.0, 1.0, "a", "b"))
        else:
            rows.append(_row(event, 0.5, 0.5, "a", "a"))
    summary = summarize_immediate_results(rows, event_set_hash="hash")
    gate = immediate_gate(summary, min_valid=6, min_nonzero=5)
    assert gate["passed"] is True
    summary["azure_used"] = True
    assert immediate_gate(summary, min_valid=6, min_nonzero=5)["passed"] is False
