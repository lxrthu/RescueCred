from pathlib import Path

import pytest

from rescuecredit.toolsandbox_router import (
    apply_flip,
    choose_flip_threshold,
    deterministic_group_folds,
    desired_candidate,
    flip_target,
    fit_logistic_head,
    router_probabilities,
)
from scripts.freeze_toolsandbox_v5_protocol import CONFIG, THRESHOLDS


def test_router_targets_only_mask_errors():
    assert desired_candidate("rescue_preference") == "b"
    assert desired_candidate("reverse_preference") == "a"
    assert flip_target("b", "rescue_preference") == 0
    assert flip_target("a", "rescue_preference") == 1
    assert flip_target("a", "reverse_preference") == 0
    assert flip_target("b", "reverse_preference") == 1
    assert apply_flip("a", True) == "b"
    assert apply_flip("b", False) == "b"


def test_threshold_selection_uses_oof_wins_and_minimum_coverage():
    audit = choose_flip_threshold(
        [0.9, 0.8, 0.2, 0.1],
        ["a", "b", "b", "a"],
        [
            "rescue_preference",
            "reverse_preference",
            "rescue_preference",
            "reverse_preference",
        ],
        [0.5, 0.85],
        min_flips=2,
    )
    assert audit["baseline_accuracy"] == 0.5
    assert audit["selected"]["threshold"] == 0.5
    assert audit["selected"]["accuracy"] == 1.0
    assert audit["selected"]["wins"] == 2
    assert audit["selected"]["losses"] == 0


def test_group_folds_are_task_disjoint_and_deterministic():
    groups = ["a", "a", "b", "c", "c", "d"]
    first = deterministic_group_folds(groups, folds=2, seed=42)
    second = deterministic_group_folds(groups, folds=2, seed=42)
    assert first == second
    for fold in first:
        validation_groups = {groups[index] for index in fold}
        training_groups = {
            groups[index] for index in range(len(groups)) if index not in set(fold)
        }
        assert not (validation_groups & training_groups)


def test_v5_protocol_freezes_independent_router_and_strict_gate():
    assert CONFIG["group_folds"] == 5
    assert CONFIG["projection_dim"] == 128
    assert THRESHOLDS["require_oof_improvement"] is True
    assert THRESHOLDS["require_rescue_noninferiority"] is True
    assert THRESHOLDS["require_reverse_improvement"] is True


def test_logistic_router_learns_a_separable_flip_rule():
    torch = pytest.importorskip("torch")
    features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    checkpoint = fit_logistic_head(
        features,
        labels,
        seed=42,
        steps=200,
        learning_rate=0.05,
        weight_decay=0.0,
    )
    probabilities = router_probabilities(features, checkpoint)
    assert torch.all(probabilities[:2] < 0.5)
    assert torch.all(probabilities[2:] > 0.5)


def test_runner_keeps_old_confirmation_posthoc():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/cloud/run_toolsandbox_v5_seed42.sh"
    ).read_text()
    assert "posthoc_confirm" in source
    assert "old V4.5 confirmation split is known" in source
    assert "check_toolsandbox_v5_gate.py" in source


def test_public_scorer_has_no_private_outcome_argument():
    source = (
        Path(__file__).resolve().parents[1] / "scripts/score_toolsandbox_v5_router.py"
    ).read_text()
    assert '"--private-outcomes"' not in source
    assert '"private_outcomes_read": False' in source


def test_unknown_causal_decision_is_rejected():
    with pytest.raises(ValueError):
        desired_candidate("zero_delta")
