from pathlib import Path

from scripts.freeze_toolsandbox_v46_protocol import CONFIG, THRESHOLDS
from scripts.train_toolsandbox_v43_preference import V46_PROTOCOL_STATUS
from scripts.train_toolsandbox_v46_residual import residual_route


def test_v46_is_residual_from_common_mask():
    assert V46_PROTOCOL_STATUS == "frozen_before_toolsandbox_v46_development_training"
    assert CONFIG["target_residual"] == 0.05
    assert CONFIG["confidence_margin"] == 0.05
    assert CONFIG["sampling"] == "all_unique_events_identical_order"


def test_v46_selectively_corrects_only_weak_or_wrong_mask_margins():
    assert residual_route("rescue_preference", 0.10, 0.05) == ("preserve", 1.0)
    assert residual_route("rescue_preference", -0.10, 0.05) == ("correct", 1.0)
    assert residual_route("reverse_preference", -0.10, 0.05) == ("preserve", -1.0)
    assert residual_route("reverse_preference", 0.10, 0.05) == ("correct", -1.0)


def test_v46_gate_requires_signed_direction_not_only_gap():
    assert THRESHOLDS["min_reverse_margin_decrease"] == 0.02
    source = (
        Path(__file__).resolve().parents[1] / "scripts/check_toolsandbox_v46_gate.py"
    ).read_text()
    assert '"signed_rescue_shift"' in source
    assert '"signed_reverse_shift"' in source
    assert '"rescue_noninferiority"' in source


def test_v46_runner_marks_old_confirmation_posthoc():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/cloud/run_toolsandbox_v46_seed42.sh"
    ).read_text()
    assert "post-hoc diagnostic" in source
    assert "posthoc_confirm" in source
    assert "check_toolsandbox_v46_gate.py" in source
