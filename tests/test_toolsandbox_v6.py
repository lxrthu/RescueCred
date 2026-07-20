from pathlib import Path

import pytest

from rescuecredit.toolsandbox_selective_router import (
    average_precision,
    calibration_metrics,
    choose_conservative_threshold,
    conservative_choice,
    fit_probe,
    probe_probabilities,
    reverse_target,
    roc_auc,
)
from scripts.freeze_toolsandbox_v6_protocol import CONFIG, GATE


def test_reverse_only_policy_defaults_to_harness_b():
    assert reverse_target("rescue_preference") == 0
    assert reverse_target("reverse_preference") == 1
    assert conservative_choice(0.89, 0.90) == "b"
    assert conservative_choice(0.90, 0.90) == "a"


def test_auc_and_pr_auc_are_exact_for_separable_scores():
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.8, 0.2, 0.9]
    assert roc_auc(labels, scores) == 1.0
    assert average_precision(labels, scores) == 1.0
    metrics = calibration_metrics(labels, scores, bins=2)
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0


def test_pr_auc_treats_tied_scores_as_one_threshold_group():
    assert average_precision([1, 0], [0.5, 0.5]) == 0.5


def test_threshold_maximizes_reverse_recall_under_rescue_budget():
    audit = choose_conservative_threshold(
        [0, 0, 1, 1],
        [0.95, 0.10, 0.90, 0.80],
        [0.75, 0.85, 0.925, 0.975],
        rescue_delta=0.0,
    )
    selected = audit["selected"]
    assert selected["threshold"] == 0.975
    assert selected["rescue_harms"] == 0
    assert selected["reverse_recall"] == 0.0


def test_threshold_allows_best_recall_when_budget_permits_one_harm():
    audit = choose_conservative_threshold(
        [0, 0, 1, 1],
        [0.95, 0.10, 0.90, 0.80],
        [0.75, 0.85, 0.925, 0.975],
        rescue_delta=0.5,
    )
    selected = audit["selected"]
    assert selected["threshold"] == 0.75
    assert selected["reverse_recall"] == 1.0
    assert selected["rescue_drop"] == 0.5


def test_semantic_probe_can_fit_a_nonlinear_reverse_rule():
    torch = pytest.importorskip("torch")
    features = torch.tensor(
        [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]]
    )
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
    checkpoint = fit_probe(
        features,
        labels,
        method="semantic_probe",
        seed=42,
        steps=500,
        learning_rate=0.03,
        weight_decay=0.0,
        hidden_dim=8,
    )
    probabilities = probe_probabilities(features, checkpoint)
    assert torch.all(probabilities[labels == 1] > 0.8)
    assert torch.all(probabilities[labels == 0] < 0.2)


def test_v6_protocol_is_conservative_and_cross_task():
    assert CONFIG["group_folds"] == 5
    assert CONFIG["rescue_delta"] == 0.02
    assert GATE["min_cross_task_roc_auc"] == 0.70
    assert GATE["require_development_rescue_noninferiority"] is True


def test_cloud_runner_reuses_frozen_v5_features_and_keeps_posthoc_separate():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/cloud/run_toolsandbox_v6_diagnostic_seed42.sh"
    ).read_text()
    assert "V5/features/train_features.pt" in source
    assert "posthoc_confirm" in source
    assert "default_b" in source
