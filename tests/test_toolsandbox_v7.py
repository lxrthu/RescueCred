import copy
from pathlib import Path

import pytest

from rescuecredit.toolsandbox_active_shadow import (
    acquisition_mask,
    active_route_metrics,
    build_active_shadow_features,
    choose_active_route_threshold,
    exact_binomial_upper_bound,
    minimum_zero_harm_calibration_size,
)
from scripts.freeze_toolsandbox_v7_protocol import CONFIG, GATE


def _row(receipt_a="missing required field", receipt_b='{"status":"ok"}'):
    branch = {
        "valid": True,
        "score": {"similarity": 1.0},
        "score_trace": [{"similarity": 1.0}],
        "progress_auc": 1.0,
        "ending_context_digest": "PROTECTED",
    }
    return {
        "action_a": {"tool": "send", "arguments": {"content": "x"}},
        "action_b": {
            "tool": "send",
            "arguments": {"content": "x", "recipient": "+123"},
        },
        "branch_a": {
            **branch,
            "receipts": [
                {
                    "action": {"tool": "send"},
                    "content": receipt_a,
                    "exception": "ValueError",
                }
            ],
        },
        "branch_b": {
            **branch,
            "receipts": [
                {
                    "action": {"tool": "send"},
                    "content": receipt_b,
                    "exception": None,
                }
            ],
        },
        "decision": "rescue_preference",
    }


def test_active_features_use_first_receipt_but_ignore_protected_outcomes():
    row = _row()
    original = build_active_shadow_features(row, hash_dimension=32)
    changed_receipt = build_active_shadow_features(
        _row(receipt_b="permission denied"), hash_dimension=32
    )
    protected = copy.deepcopy(row)
    protected["decision"] = "reverse_preference"
    protected["branch_a"]["score"] = {"similarity": -999.0}
    protected["branch_a"]["score_trace"] = [{"similarity": -999.0}]
    protected["branch_a"]["progress_auc"] = -999.0
    protected["branch_a"]["ending_context_digest"] = "CHANGED"
    assert original != changed_receipt
    assert original == build_active_shadow_features(protected, hash_dimension=32)


def test_active_features_reject_protected_keys_embedded_in_receipt_content():
    row = _row(receipt_b='{"score_trace":[0.5, 1.0]}')
    with pytest.raises(ValueError, match="protected keys"):
        build_active_shadow_features(row, hash_dimension=32)


def test_active_features_reject_casefolded_protected_exception_keys():
    row = _row()
    row["branch_a"]["receipts"][0]["exception"] = {
        "Official_Score": 0.9,
        "Ground_Truth": "A",
    }
    with pytest.raises(ValueError, match="protected keys"):
        build_active_shadow_features(row, hash_dimension=32)


def test_exact_binomial_bound_exposes_current_sample_limit():
    assert exact_binomial_upper_bound(0, 41, alpha=0.05) > 0.02
    required = minimum_zero_harm_calibration_size(0.02, alpha=0.05)
    assert required == 149
    assert exact_binomial_upper_bound(0, required, alpha=0.05) <= 0.02


def test_acquisition_mask_obeys_probe_budget_deterministically():
    mask, threshold = acquisition_mask(
        [0.1, 0.9, 0.8, 0.2], ["a", "b", "c", "d"], max_probe_rate=0.5
    )
    assert mask == [False, True, True, False]
    assert threshold == 0.8


def test_active_route_defaults_unprobed_and_low_confidence_events_to_b():
    metrics = active_route_metrics(
        [0, 0, 1, 1],
        [0.95, 0.1, 0.9, 0.8],
        [False, True, True, False],
        route_threshold=0.85,
        alpha=0.05,
    )
    assert metrics["probe_rate"] == 0.5
    assert metrics["route_to_a"] == 1
    assert metrics["reverse_recall"] == 0.5
    assert metrics["rescue_drop"] == 0.0


def test_route_threshold_maximizes_recall_under_empirical_rescue_budget():
    audit = choose_active_route_threshold(
        [0, 0, 1, 1],
        [0.7, 0.1, 0.9, 0.8],
        [True, True, True, True],
        [0.75, 0.85, 0.925, 0.975],
        rescue_delta=0.0,
        alpha=0.05,
    )
    assert audit["selected"]["route_threshold"] == 0.75
    assert audit["selected"]["reverse_recall"] == 1.0
    assert audit["selected"]["rescue_harms"] == 0
    assert audit["certified_selected"] is None


def test_v7_protocol_freezes_requested_feasibility_gate():
    assert CONFIG["max_probe_rate"] == 0.30
    assert CONFIG["rescue_delta"] == 0.02
    assert GATE["min_active_cross_task_roc_auc"] == 0.75
    assert GATE["min_reverse_recall"] == 0.20


def test_v7_runner_reuses_receipts_without_full_shadow_rerun():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/cloud/run_toolsandbox_v7_active_shadow_seed42.sh"
    ).read_text(encoding="utf-8")
    assert "full_offset85_h8/candidate_events.jsonl" in source
    assert "audit_toolsandbox_v44_candidates.py" not in source
    assert "run_toolsandbox_v7_active_shadow" not in source


def test_v7_training_uses_nested_task_calibration_and_untouched_evaluation():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/train_toolsandbox_v7_active_shadow.py"
    ).read_text(encoding="utf-8")
    assert "model_indices" in source
    assert "calibration_indices" in source
    assert "evaluation_indices" in source
    assert '"deployment_ready": False' in source
