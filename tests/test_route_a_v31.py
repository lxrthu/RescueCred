from scripts.check_route_a_v31_gate import build_gate
from scripts.freeze_route_a_v31_protocol import GATE_THRESHOLDS


def _artifacts():
    mask = {
        "method": "mask",
        "validation_file_sha256": "validation",
        "validity_first_accuracy": 0.8,
        "missing_required_accuracy": 0.9,
        "both_valid_causal_accuracy": 0.5,
        "run_summary_sha256": "mask-run",
        "adapter_sha256": "mask-adapter",
        "base_model_sha256": "base",
    }
    mask_run = {
        "method": "mask",
        "presentations_per_epoch": 255,
        "active_event_presentations": 765,
        "_actual_run_summary_sha256": "mask-run",
        "adapter_sha256": "mask-adapter",
    }
    v31 = {
        "method": "v31",
        "validation_file_sha256": "validation",
        "validity_first_events": 30,
        "validity_first_accuracy": 0.9,
        "missing_required_accuracy": 1.0,
        "both_valid_causal_events": 8,
        "both_valid_causal_accuracy": 0.75,
        "run_summary_sha256": "v31-run",
        "adapter_sha256": "v31-adapter",
        "base_model_sha256": "base",
    }
    v31_run = {
        "method": "v31",
        "presentations_per_epoch": 255,
        "active_event_presentations": 765,
        "validity_first": True,
        "causal_class_balanced": False,
        "presented_decisions": {
            "validity_b_over_a": 500,
            "rescue_preference": 200,
            "reverse_preference": 65,
        },
        "run_summary_sha256": "v31-run",
        "adapter_sha256": "v31-adapter",
        "protocol_lock_sha256": "lock",
    }
    protocol = {
        "validation_sha256": "validation",
        "gate_thresholds": GATE_THRESHOLDS,
        "mask_run_sha256": "mask-run",
        "mask_adapter_sha256": "mask-adapter",
        "mask_eval_sha256": "mask-eval",
        "_actual_protocol_lock_sha256": "lock",
        "source_identity_matches": True,
        "expected_presented_decisions": {
            "validity_b_over_a": 500,
            "rescue_preference": 200,
            "reverse_preference": 65,
        },
        "base_model_sha256": "base",
    }
    mask["_actual_eval_sha256"] = "mask-eval"
    return mask, mask_run, v31, v31_run, protocol


def test_v31_gate_passes_validity_first_and_both_valid_improvement():
    gate = build_gate(*_artifacts())
    assert gate["passed"] is True
    assert gate["both_valid_accuracy_improvement"] == 0.25


def test_v31_gate_rejects_missing_argument_regression():
    artifacts = list(_artifacts())
    artifacts[2]["missing_required_accuracy"] = 0.5
    gate = build_gate(*artifacts)
    assert gate["passed"] is False
    assert gate["checks"]["missing_required_is_safe"] is False


def test_v31_gate_rejects_changed_mask_eval_hash():
    artifacts = list(_artifacts())
    artifacts[0]["_actual_eval_sha256"] = "changed"
    gate = build_gate(*artifacts)
    assert gate["passed"] is False
    assert gate["checks"]["mask_eval_frozen"] is False


def test_v31_gate_rejects_changed_source_identity():
    artifacts = list(_artifacts())
    artifacts[4]["source_identity_matches"] = False
    gate = build_gate(*artifacts)
    assert gate["passed"] is False
    assert gate["checks"]["source_identity_frozen"] is False


def test_v31_gate_rejects_nonfrozen_sampling_counts():
    artifacts = list(_artifacts())
    artifacts[3]["presented_decisions"] = {
        "validity_b_over_a": 499,
        "rescue_preference": 201,
        "reverse_preference": 65,
    }
    gate = build_gate(*artifacts)
    assert gate["passed"] is False
    assert gate["checks"]["natural_ratio_presentations_exact"] is False
