import json

import pytest

from scripts.analyze_harness_blindness import analyze, training_signal


def _arm(s_on, s_off, first_pass, intervention_rate, signal):
    return {
        "run": {
            "audit_stats": {"valid_audits": 8},
            "shadow_steps": 20,
        },
        "eval": {
            "s_on": s_on,
            "s_off": s_off,
            "dependence_gap": s_on - s_off,
            "first_pass": first_pass,
            "first_attempt_accuracy": first_pass,
            "intervention_rate": intervention_rate,
            "rescued_tasks": int(s_on > s_off),
        },
        "training_signal": signal,
        "curve": [],
    }


def test_training_signal_identifies_hidden_rescue_without_negative_naive_credit(tmp_path):
    rows = [
        {
            "assisted_returns": [1.0, 0.0, 1.0],
            "diagnostic_shadow_returns": [0.0, 0.0, 1.0],
            "diagnostic_replay_valid": [True, True, True],
            "prefix_assigned_advantages": [0.5, -0.5, 0.0],
            "g0_hat": [1.0, 0.0, 1.0],
            "loss_corr": 0.0,
            "weighted_loss_corr": 0.0,
        }
    ]
    path = tmp_path / "train_rank0.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    signal = training_signal(tmp_path)
    assert signal["rescued"]["events"] == 1
    assert signal["rescued"]["negative_prefix_rate"] == 0.0
    assert signal["zero_delta"]["events"] == 2


def test_gate_separates_mechanism_from_method_success():
    protocol = {
        "scope": "controlled",
        "gate": {
            "min_diagnostic_rescue_events": 5,
            "max_naive_negative_prefix_rate_on_rescues": 0.0,
        },
    }
    naive_signal = {
        "rescued": {
            "events": 8,
            "negative_prefix_rate": 0.0,
            "mean_prefix_assigned_advantage": 0.2,
        },
        "nonzero_correction_updates": 0,
        "shadow_alignment_checks": 8,
        "shadow_alignment_rate": 0.0,
    }
    mask_signal = {
        "rescued": {
            "events": 8,
            "negative_prefix_rate": 0.0,
            "zero_prefix_rate": 1.0,
            "mean_prefix_assigned_advantage": 0.0,
        },
        "nonzero_correction_updates": 3,
        "shadow_alignment_checks": 8,
        "shadow_alignment_rate": 0.0,
    }
    rescue_signal = {
        "rescued": {
            "events": 8,
            "negative_prefix_rate": 0.5,
            "zero_prefix_rate": 0.0,
            "mean_prefix_assigned_advantage": -0.2,
        },
        "nonzero_correction_updates": 3,
        "shadow_alignment_checks": 8,
        "shadow_alignment_rate": 1.0,
    }
    arms = {
        "naive_h_grpo": _arm(0.7, 0.3, 0.4, 0.6, naive_signal),
        "mask_correction": _arm(0.7, 0.4, 0.5, 0.5, mask_signal),
        "rescuecredit": _arm(0.75, 0.5, 0.6, 0.4, rescue_signal),
    }
    result = analyze(protocol, {}, arms)
    assert result["mechanism_supported"] is True
    assert result["method_supported"] is True
    assert result["passed"] is True

    arms["rescuecredit"]["eval"]["s_off"] = 0.3
    result = analyze(protocol, {}, arms)
    assert result["mechanism_supported"] is True
    assert result["method_supported"] is False
    assert result["passed"] is False


def test_training_signal_rejects_duplicate_or_forged_weighted_rows(tmp_path):
    base = {
        "rank": 0,
        "update": 1,
        "method": "mask_correction",
        "run_id": "mask_correction_seed42",
        "config_hash": "cfg",
        "realized_group_size": 4,
        "loss_corr": 2.0,
        "weighted_loss_corr": 0.2,
        "assisted_returns": [],
        "diagnostic_shadow_returns": [],
        "diagnostic_replay_valid": [],
        "prefix_assigned_advantages": [],
        "g0_hat": [],
    }
    path = tmp_path / "train_rank0.jsonl"
    path.write_text(json.dumps(base) + "\n" + json.dumps(base) + "\n")
    with pytest.raises(ValueError, match="duplicate training row"):
        training_signal(
            tmp_path,
            expected_method="mask_correction",
            expected_config_hash="cfg",
            expected_run_id="mask_correction_seed42",
            expected_lambda_corr=0.1,
        )

    forged = dict(base, weighted_loss_corr=1.0)
    path.write_text(json.dumps(forged) + "\n")
    with pytest.raises(ValueError, match="weighted correction loss mismatch"):
        training_signal(
            tmp_path,
            expected_method="mask_correction",
            expected_config_hash="cfg",
            expected_run_id="mask_correction_seed42",
            expected_lambda_corr=0.1,
        )
