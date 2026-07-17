from rescuecredit.route_a_task_eval import (
    event_set_hash,
    paired_gate,
    summarize_task_results,
    validated_action,
)
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validated_action_rejects_malformed_and_accepts_rest_calls():
    assert validated_action({"tool": "post:/mail/send", "arguments": {"x": 1}}) == {
        "tool": "post:/mail/send",
        "arguments": {"x": 1},
    }
    assert validated_action({"tool": "bad", "arguments": {}}) is None
    assert validated_action({"tool": "mail__send", "arguments": []}) is None


def test_event_set_hash_is_order_independent():
    assert event_set_hash([{"event_id": "b"}, {"event_id": "a"}]) == event_set_hash(
        [{"event_id": "a"}, {"event_id": "b"}]
    )


def test_summary_uses_official_scores_and_tracks_failures():
    rows = [
        {
            "evaluation_valid": True,
            "official_score": 1.0,
            "correction_matches_reference": True,
            "generation_failed": False,
            "candidate_execution_failed": False,
        },
        {
            "evaluation_valid": True,
            "official_score": 0.5,
            "correction_matches_reference": False,
            "generation_failed": True,
            "candidate_execution_failed": True,
        },
    ]
    summary = summarize_task_results(rows, method="v2", events_hash="h")
    assert summary["mean_official_requirement_score"] == 0.75
    assert summary["task_success_rate"] == 0.5
    assert summary["exact_correction_rate"] == 0.5
    assert summary["generation_failure_rate"] == 0.5


def test_paired_gate_requires_score_gain_without_task_success_regression():
    mask_rows = [
        {"event_id": "a", "evaluation_valid": True, "official_score": 0.2},
        {"event_id": "b", "evaluation_valid": True, "official_score": 1.0},
    ]
    v2_rows = [
        {"event_id": "a", "evaluation_valid": True, "official_score": 0.5},
        {"event_id": "b", "evaluation_valid": True, "official_score": 1.0},
    ]
    mask = {
        "event_set_hash": "h",
        "mean_official_requirement_score": 0.6,
        "task_success_rate": 0.5,
        "reference_free_model_inputs": True,
        "reference_suffix_used": False,
        "adapter_scoring_failure_rate": 0.0,
    }
    v2 = {
        "event_set_hash": "h",
        "mean_official_requirement_score": 0.75,
        "task_success_rate": 0.5,
        "reference_free_model_inputs": True,
        "reference_suffix_used": False,
        "adapter_scoring_failure_rate": 0.0,
    }
    gate = paired_gate(mask, v2, mask_rows, v2_rows, min_events=2)
    assert gate["passed"] is True
    assert gate["v2_better_events"] == 1
    assert gate["v2_worse_events"] == 0


def test_dev_protocol_scores_candidates_and_never_replays_reference_suffix():
    evaluator = (ROOT / "scripts/evaluate_route_a_appworld_dev.py").read_text(
        encoding="utf-8"
    )
    runner = (
        ROOT / "scripts/cloud/run_route_a_appworld_dev_pair.sh"
    ).read_text(encoding="utf-8")
    assert "AdapterScorerWorker" in evaluator
    assert "ContinuationWorker" in evaluator
    assert "calls[call_index + 1" not in evaluator
    assert '"reference_suffix_used": False' in evaluator
    assert "route_a_adapter_scorer_worker.py" in runner
    assert "appworld_azure_continuation_worker.py" in runner
