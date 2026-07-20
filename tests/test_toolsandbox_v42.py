from collections import Counter
from pathlib import Path

from rescuecredit.toolsandbox_preference import training_preference
from scripts.check_toolsandbox_v42_gate import build_gate
from scripts.freeze_toolsandbox_v42_protocol import (
    CONFIG,
    CONFIRMATION_THRESHOLDS,
    DEVELOPMENT_THRESHOLDS,
)
from scripts.train_route_a_preference import balanced_causal_epoch_order


def _train_row(event_id: str, decision: str) -> dict:
    return {
        "event_id": event_id,
        "replay_valid": True,
        "decision": decision,
        "prompt": "visible only",
        "action_a": {"tool": "send", "arguments": {"to": "a"}},
        "action_b": {
            "tool": "send",
            "arguments": {"to": "a", "body": "hello"},
        },
    }


def test_v42_epoch_is_exactly_balanced_and_deterministic():
    rows = [
        *[_train_row(f"r{index}", "rescue_preference") for index in range(33)],
        *[_train_row(f"x{index}", "reverse_preference") for index in range(3)],
    ]
    first = balanced_causal_epoch_order(rows, 42, 0, 36)
    second = balanced_causal_epoch_order(rows, 42, 0, 36)
    assert [row["event_id"] for row in first] == [
        row["event_id"] for row in second
    ]
    assert Counter(row["decision"] for row in first) == {
        "rescue_preference": 18,
        "reverse_preference": 18,
    }
    reverse_counts = Counter(
        row["event_id"] for row in first if row["decision"] == "reverse_preference"
    )
    assert set(reverse_counts.values()) == {6}


def test_mask_and_v42_share_events_but_use_different_reverse_label():
    reverse = _train_row("x", "reverse_preference")
    mask_chosen, _, _ = training_preference(reverse, "mask")
    v42_chosen, _, _ = training_preference(reverse, "v4")
    assert mask_chosen == reverse["action_b"]
    assert v42_chosen == reverse["action_a"]


def test_v42_frozen_config_changes_only_sampling_and_strength():
    assert CONFIG["presentations_per_epoch"] == 36
    assert CONFIG["epochs"] == 3
    assert CONFIG["absolute_margin_coef"] == 1.0
    assert CONFIG["target_margin"] == 0.05
    assert CONFIG["sampling"] == "identical_class_balanced_rescue_reverse"
    assert DEVELOPMENT_THRESHOLDS["min_selection_disagreements"] == 1
    assert CONFIRMATION_THRESHOLDS["min_selection_disagreements"] == 3
    assert CONFIRMATION_THRESHOLDS["min_causal_accuracy_improvement"] == 0.05


def _eval(method: str, accuracy: float) -> dict:
    return {
        "method": method,
        "evaluation_role": "confirmation",
        "event_set_hash": "events",
        "events": 20,
        "valid_events": 20,
        "decisions": {"rescue_preference": 17, "reverse_preference": 3},
        "causal_accuracy": accuracy,
        "rescue_accuracy": 1.0,
        "reverse_accuracy": 0.0 if method == "mask" else 1.0,
        "selected_b_rate": 1.0 if method == "mask" else 0.85,
        "mean_selected_terminal_similarity": 0.5,
        "mean_selected_progress_auc": 0.5,
        "worker_receives_public_prompt_and_candidates_only": True,
        "offline_outcomes_joined_after_scoring": True,
    }


def _rows(method: str) -> list[dict]:
    return [
        {
            "event_id": f"e{index}",
            "selected": (
                "a" if method == "v42" and index < 3 else "b"
            ),
            "decision": (
                "reverse_preference" if index < 3 else "rescue_preference"
            ),
            "replay_valid": True,
            "causal_correct": method == "v42" or index >= 3,
            "selected_terminal_similarity": 0.5,
            "selected_progress_auc": 0.5,
        }
        for index in range(20)
    ]


def test_confirmation_gate_recomputes_rows_and_requires_real_flips():
    protocol = {
        "train_sha256": "train",
        "train_events": 36,
        "config": CONFIG,
        "expected_presented_event_sequence_sha256": "sequence",
        "expected_presented_source_decisions": {
            "rescue_preference": 54,
            "reverse_preference": 54,
        },
        "expected_presented_decisions": {
            "mask": {"b_over_a": 108},
            "v42": {"a_over_b": 54, "b_over_a": 54},
        },
        "gate_thresholds": {
            "development": DEVELOPMENT_THRESHOLDS,
            "confirmation": CONFIRMATION_THRESHOLDS,
        },
        "scope": "test",
    }
    common_run = {
        "train_file_sha256": "train",
        "presentations_per_epoch": 36,
        "active_event_presentations": 108,
        "presented_event_sequence_sha256": "sequence",
        "presented_source_decisions": {
            "rescue_preference": 54,
            "reverse_preference": 54,
        },
        "absolute_margin_coef": 1.0,
        "target_margin": 0.05,
        "loss_definition": "unit_weight*(dpo_shift+absolute_margin)",
    }
    mask_run = {
        **common_run,
        "method": "mask",
        "presented_decisions": {"b_over_a": 108},
    }
    v42_run = {
        **common_run,
        "method": "v42",
        "presented_decisions": {"a_over_b": 54, "b_over_a": 54},
    }
    gate = build_gate(
        role="confirmation",
        mask_eval=_eval("mask", 0.85),
        v42_eval=_eval("v42", 1.0),
        mask_run=mask_run,
        v42_run=v42_run,
        mask_rows=_rows("mask"),
        v42_rows=_rows("v42"),
        protocol=protocol,
        eval_manifest={
            "event_set_hash": "events",
            "role": "evaluation",
            "protected_outcomes_in_prompt": False,
            "official_branch_metrics_in_training_file": False,
            "branch_receipts_exported": False,
            "reference_actions_read_or_exported": False,
            "events": 20,
        },
        eval_audit={
            "protocol_validated": True,
            "harness_interface": "tool_id_v2",
            "controlled": {"nonzero_events": 20},
            "natural": {"nonzero_events": 0},
        },
        eval_audit_gate={"mechanism_passed": True},
        identity={"artifacts_bound": True},
    )
    assert gate["passed"] is True
    assert gate["selection_disagreements"] == 3
    assert gate["v42_wins"] == 3
    assert gate["v42_losses"] == 0


def test_runner_freezes_offset165_before_training_and_gates_api_calls():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "cloud"
        / "run_toolsandbox_v42_seed42.sh"
    ).read_text(encoding="utf-8")
    freeze_confirmation = source.index("--scenario-offset 165")
    freeze_training = source.index(
        '"$MODEL_PY" scripts/freeze_toolsandbox_v42_protocol.py'
    )
    train = source.index(
        'CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" '
        "scripts/train_toolsandbox_v42_preference.py"
    )
    development_gate = source.index("--role development")
    confirmation_audit = source.index(
        '"$APP_PY" scripts/audit_toolsandbox_signal.py', development_gate
    )
    confirmation_gate = source.index("--role confirmation")
    assert freeze_confirmation < freeze_training < train < development_gate
    assert development_gate < confirmation_audit < confirmation_gate
    assert source.count("--exclude-protocol") == 4
    assert "--absolute-margin-coef 1.0 --target-margin 0.05" in source
    assert "--presentations-per-epoch 36" in source
