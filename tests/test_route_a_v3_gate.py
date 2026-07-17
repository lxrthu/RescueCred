from rescuecredit.frozen_bank import directory_sha256
from scripts.check_route_a_v3_gate import EXPECTED_CONFIG, build_gate


def summary(accuracy, reverse, margin, *, base_model="base"):
    return {
        "validation_file_sha256": "same",
        "causal_events": 7,
        "causal_accuracy": accuracy,
        "reverse_accuracy": reverse,
        "rescue_accuracy": 2 / 3,
        "mean_signed_causal_margin": margin,
        "method": "v3",
        "adapter_sha256": "adapter",
        "run_summary_sha256": "run-summary",
        "base_model_sha256": base_model,
    }


def test_v3_gate_requires_accuracy_margin_and_reverse_improvements(tmp_path):
    mask = summary(3 / 7, 1 / 4, 0.0)
    mask["method"] = "mask"
    run = {
        **EXPECTED_CONFIG,
        "train_file_sha256": "train",
        "adapter_sha256": "adapter",
        "protocol_lock_sha256": "protocol",
        "base_model_sha256": "base",
        "presentations_per_epoch": 86,
        "active_event_presentations": 258,
        "presentation_budget_matches_mask": True,
        "zero_delta_rows_excluded_from_causal_loss": True,
        "presented_decisions": {
            "rescue_preference": 129,
            "reverse_preference": 129,
        },
    }
    protocol = {
        "status": "frozen_before_v3_outcomes",
        "train_sha256": "train",
        "validation_sha256": "same",
        "mask_eval_sha256": "mask-eval",
        "mask_run_sha256": "mask-run",
        "mask_adapter_sha256": "adapter",
        "base_model_sha256": "base",
        "gate_thresholds": {
            "min_causal_events": 5,
            "min_accuracy_improvement": 0.10,
            "require_positive_mean_signed_margin": True,
            "require_reverse_accuracy_improvement": True,
        },
    }
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights").write_text("base")
    mask_run = {
        "method": "mask",
        "adapter": str(tmp_path / "mask_adapter"),
        "model": str(model_dir),
    }
    (tmp_path / "mask_adapter").mkdir()
    (tmp_path / "mask_adapter" / "weights").write_text("x")
    protocol["mask_adapter_sha256"] = directory_sha256(tmp_path / "mask_adapter")
    protocol["base_model_sha256"] = directory_sha256(model_dir)
    run["base_model_sha256"] = protocol["base_model_sha256"]
    mask["base_model_sha256"] = protocol["base_model_sha256"]
    mask["adapter_sha256"] = protocol["mask_adapter_sha256"]
    mask["run_summary_sha256"] = "mask-run"
    passing_v3 = summary(4 / 7, 2 / 4, 0.02)
    passing_v3["base_model_sha256"] = protocol["base_model_sha256"]
    passing = build_gate(
        mask,
        passing_v3,
        run,
        protocol,
        mask_run=mask_run,
        mask_run_sha256="mask-run",
        mask_eval_sha256="mask-eval",
        run_summary_sha256="run-summary",
        protocol_lock_sha256="protocol",
        base_model_sha256=protocol["base_model_sha256"],
    )
    assert passing["passed"] is True
    assert all(passing["checks"].values())

    failing_v3 = summary(4 / 7, 1 / 4, 0.02)
    failing_v3["base_model_sha256"] = protocol["base_model_sha256"]
    no_reverse_gain = build_gate(
        mask,
        failing_v3,
        run,
        protocol,
        mask_run=mask_run,
        mask_run_sha256="mask-run",
        mask_eval_sha256="mask-eval",
        run_summary_sha256="run-summary",
        protocol_lock_sha256="protocol",
        base_model_sha256=protocol["base_model_sha256"],
    )
    assert no_reverse_gain["passed"] is False
    assert no_reverse_gain["checks"]["v3_improves_reverse_accuracy"] is False
