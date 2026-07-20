from pathlib import Path

from scripts.freeze_toolsandbox_v45_learner_protocol import (
    CONFIG,
    CONFIRMATION_THRESHOLDS,
    DEVELOPMENT_THRESHOLDS,
    SOURCE_PATHS,
)
from scripts.train_route_a_preference import balanced_causal_epoch_order
from scripts.train_toolsandbox_v43_preference import V45_PROTOCOL_STATUS


def _rows(rescue: int = 41, reverse: int = 85):
    return [
        {"event_id": f"r{i}", "decision": "rescue_preference"}
        for i in range(rescue)
    ] + [
        {"event_id": f"x{i}", "decision": "reverse_preference"}
        for i in range(reverse)
    ]


def test_v45_balanced_schedule_matches_frozen_v44_pool():
    rows = _rows()
    for epoch in range(3):
        ordered = balanced_causal_epoch_order(rows, 42, epoch, 126)
        assert len(ordered) == 126
        assert sum(row["decision"] == "rescue_preference" for row in ordered) == 63
        assert sum(row["decision"] == "reverse_preference" for row in ordered) == 63
    assert CONFIG["presentations_per_epoch"] == 126
    assert CONFIG["reference_anchor_coef"] == 0.25


def test_v45_thresholds_are_stronger_on_confirmation():
    assert DEVELOPMENT_THRESHOLDS["min_eval_events"] == 40
    assert DEVELOPMENT_THRESHOLDS["min_eval_rescue_events"] == 5
    assert DEVELOPMENT_THRESHOLDS["min_eval_reverse_events"] == 5
    assert CONFIRMATION_THRESHOLDS["min_causal_accuracy_improvement"] == 0.05
    assert V45_PROTOCOL_STATUS == "frozen_before_toolsandbox_v45_training_and_eval_outcomes"


def test_v45_source_inventory_binds_learner_and_runner():
    assert "scripts/check_toolsandbox_v45_gate.py" in SOURCE_PATHS
    assert "scripts/cloud/run_toolsandbox_v45_seed42.sh" in SOURCE_PATHS
    assert "refine-logs/TOOLSANDBOX_V45_PLAN.md" in SOURCE_PATHS


def test_v45_runner_freezes_both_evaluation_sets_before_outcomes_and_stops_on_dev():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "cloud"
        / "run_toolsandbox_v45_seed42.sh"
    ).read_text(encoding="utf-8")
    freeze_dev = source.index("freeze_candidate development 125")
    freeze_confirm = source.index("freeze_candidate confirmation 165")
    freeze_learner = source.index(
        '"$MODEL_PY" scripts/freeze_toolsandbox_v45_learner_protocol.py'
    )
    run_dev = source.index("run_candidate_audit development")
    run_confirm = source.index("run_candidate_audit confirmation")
    assert freeze_dev < freeze_confirm < freeze_learner < run_dev < run_confirm
    assert "TOOLSANDBOX_V45_FINISHED_BEFORE_CONFIRMATION" in source
    assert "--data-role evaluation" in source


def test_v45_does_not_touch_final_offset205_holdout():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "cloud"
        / "run_toolsandbox_v45_seed42.sh"
    ).read_text(encoding="utf-8")
    assert "scenario-offset 205" not in source
    assert "freeze_candidate development 125" in source
    assert "freeze_candidate confirmation 165" in source
