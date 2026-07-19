from pathlib import Path

from rescuecredit.toolsandbox_preference import (
    event_set_hash,
    matched_epoch_order,
    public_preference_prompt,
    summarize_evaluation_rows,
    training_preference,
)
from scripts.check_toolsandbox_v41_preference_gate import build_gate
from scripts.freeze_toolsandbox_v41_preference_protocol import GATE_THRESHOLDS


def _row(event_id="e0", decision="rescue_preference"):
    return {
        "event_id": event_id,
        "replay_valid": True,
        "decision": decision,
        "action_a": {"tool": "send", "arguments": {"to": "a"}},
        "action_b": {
            "tool": "send",
            "arguments": {"to": "a", "body": "hello"},
        },
    }


def test_mask_and_v4_use_same_pair_but_reverse_only_v4_direction():
    reverse = _row(decision="reverse_preference")
    mask_chosen, mask_rejected, mask_weight = training_preference(reverse, "mask")
    v4_chosen, v4_rejected, v4_weight = training_preference(reverse, "v4")
    assert mask_chosen == reverse["action_b"]
    assert mask_rejected == reverse["action_a"]
    assert v4_chosen == reverse["action_a"]
    assert v4_rejected == reverse["action_b"]
    assert mask_weight == v4_weight == 1.0


def test_preference_rejects_zero_or_invalid_credit():
    zero = _row(decision="zero_delta")
    try:
        training_preference(zero, "v4")
    except ValueError as error:
        assert "nonzero" in str(error)
    else:
        raise AssertionError("zero credit must not enter training")
    zero["decision"] = "rescue_preference"
    zero["replay_valid"] = False
    try:
        training_preference(zero, "mask")
    except ValueError as error:
        assert "replay-valid" in str(error)
    else:
        raise AssertionError("invalid replay must not enter training")


def test_public_prompt_contains_only_declared_public_inputs():
    prompt = public_preference_prompt(
        visible_history=[{"sender": "user", "content": "send mail"}],
        public_tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "send",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        action_a=_row()["action_a"],
        action_b=_row()["action_b"],
    )
    assert "send mail" in prompt
    assert "candidate_a" in prompt and "candidate_b" in prompt
    assert "decision" not in prompt
    assert "similarity" not in prompt
    assert "milestone" not in prompt
    assert "minefield" not in prompt


def test_matched_order_and_event_hash_are_deterministic():
    rows = [_row(f"e{index}") for index in range(6)]
    assert matched_epoch_order(rows, 42, 0) == matched_epoch_order(rows, 42, 0)
    assert event_set_hash(rows) == event_set_hash(list(reversed(rows)))


def test_evaluation_summary_separates_terminal_and_progress():
    rows = [
        {
            "event_id": "a",
            "replay_valid": True,
            "decision": "rescue_preference",
            "causal_correct": True,
            "selected": "b",
            "selected_terminal_similarity": 0.0,
            "selected_progress_auc": 0.5,
        },
        {
            "event_id": "b",
            "replay_valid": True,
            "decision": "reverse_preference",
            "causal_correct": False,
            "selected": "b",
            "selected_terminal_similarity": 1.0,
            "selected_progress_auc": 1.0,
        },
    ]
    summary = summarize_evaluation_rows(rows)
    assert summary["causal_accuracy"] == 0.5
    assert summary["mean_selected_terminal_similarity"] == 0.5
    assert summary["mean_selected_progress_auc"] == 0.75


def _eval(method, accuracy, terminal=0.5, progress=0.5):
    return {
        "method": method,
        "event_set_hash": "events",
        "events": 20,
        "valid_events": 20,
        "decisions": {"rescue_preference": 17, "reverse_preference": 3},
        "causal_accuracy": accuracy,
        "rescue_accuracy": 1.0,
        "reverse_accuracy": 0.0 if method == "mask" else 1.0,
        "selected_b_rate": 1.0 if method == "mask" else 0.85,
        "mean_selected_terminal_similarity": terminal,
        "mean_selected_progress_auc": progress,
        "worker_receives_public_prompt_and_candidates_only": True,
        "offline_outcomes_joined_after_scoring": True,
    }


def _selection_rows(method):
    selected = {
        "mask": ["b", "b", "b"] + ["b"] * 17,
        "v4": ["a", "a", "a"] + ["b"] * 17,
    }[method]
    return [
        {
            "event_id": f"e{index}",
            "selected": choice,
            "decision": "reverse_preference" if index < 3 else "rescue_preference",
            "replay_valid": True,
            "causal_correct": True if method == "v4" else index >= 3,
            "selected_terminal_similarity": 0.5,
            "selected_progress_auc": 0.5,
        }
        for index, choice in enumerate(selected)
    ]


def test_comparison_gate_requires_causal_gain_and_no_official_regression():
    protocol = {
        "train_sha256": "train",
        "train_events": 30,
        "config": {"epochs": 3},
        "expected_presented_event_sequence_sha256": "sequence",
        "evaluation_scenario_identity": {"fresh_hashes": ["fresh"]},
        "evaluation_protocol_sha256": "eval-lock",
        "gate_thresholds": GATE_THRESHOLDS,
        "scope": "test",
    }
    runs = {
        method: {
            "method": method,
            "train_file_sha256": "train",
            "presentations_per_epoch": 30,
            "active_event_presentations": 90,
            "presented_event_sequence_sha256": "sequence",
        }
        for method in ("mask", "v4")
    }
    gate = build_gate(
        mask_eval=_eval("mask", 0.85),
        v4_eval=_eval("v4", 1.0),
        mask_run=runs["mask"],
        v4_run=runs["v4"],
        mask_rows=_selection_rows("mask"),
        v4_rows=_selection_rows("v4"),
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
            "selected_scenario_hashes": ["fresh"],
            "protocol_validated": True,
            "harness_interface": "tool_id_v2",
            "protocol_lock_sha256": "eval-lock",
            "controlled": {"nonzero_events": 20},
            "natural": {"nonzero_events": 0},
        },
        eval_audit_gate={"mechanism_passed": True},
        identity={"artifacts": True},
    )
    assert gate["passed"] is True
    harmed = build_gate(
        mask_eval=_eval("mask", 0.85, terminal=0.5),
        v4_eval=_eval("v4", 1.0, terminal=0.4),
        mask_run=runs["mask"],
        v4_run=runs["v4"],
        mask_rows=_selection_rows("mask"),
        v4_rows=_selection_rows("v4"),
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
            "selected_scenario_hashes": ["fresh"],
            "protocol_validated": True,
            "harness_interface": "tool_id_v2",
            "protocol_lock_sha256": "eval-lock",
            "controlled": {"nonzero_events": 20},
            "natural": {"nonzero_events": 0},
        },
        eval_audit_gate={"mechanism_passed": True},
        identity={"artifacts": True},
    )
    assert harmed["passed"] is False
    assert harmed["outcome_checks"]["terminal_noninferiority"] is False


def test_runner_freezes_eval_before_training_and_audits_after_training():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "cloud"
        / "run_toolsandbox_v41_preference_seed42.sh"
    ).read_text(encoding="utf-8")
    freeze_eval = source.index('"$APP_PY" scripts/freeze_toolsandbox_v4_protocol.py')
    freeze_preference = source.index(
        '"$MODEL_PY" scripts/freeze_toolsandbox_v41_preference_protocol.py'
    )
    train = source.index(
        'CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" '
        'scripts/train_toolsandbox_v41_preference.py'
    )
    audit = source.index('"$APP_PY" scripts/audit_toolsandbox_signal.py')
    assert freeze_eval < freeze_preference < train < audit
    assert "--scenario-offset 125" in source
    assert source.count("--exclude-protocol") == 3
