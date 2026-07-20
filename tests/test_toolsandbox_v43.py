import hashlib
from pathlib import Path

from rescuecredit.toolsandbox_preference import summarize_evaluation_rows
from scripts.audit_toolsandbox_signal import _paired_row
from scripts.check_toolsandbox_v43_gate import build_gate
from scripts.freeze_toolsandbox_v43_protocol import (
    CONFIG,
    CONFIRMATION_THRESHOLDS,
    DEVELOPMENT_THRESHOLDS,
)
from scripts.prepare_toolsandbox_v43_training_data import THRESHOLDS


def test_multi_prefix_nonce_keeps_old_ids_and_makes_new_ids_unique():
    branch = {"valid": True, "score": {"similarity": 0.0}}
    common = dict(
        mode="controlled_missing_argument",
        scenario_name="scenario",
        task_hash="task",
        action_a={"tool": "a", "arguments": {}},
        action_b={"tool": "b", "arguments": {}},
        branch_a=branch,
        branch_b=branch,
        metadata={},
        credit_mode="terminal",
        horizon=4,
    )
    old = _paired_row(**common)
    first = _paired_row(**common, event_nonce="prefix=0:controlled")
    second = _paired_row(**common, event_nonce="prefix=1:controlled")
    expected_old = hashlib.sha256(
        b"controlled_missing_argument\0scenario"
    ).hexdigest()
    assert old["event_id"] == expected_old
    assert len({old["event_id"], first["event_id"], second["event_id"]}) == 3


def test_v43_frozen_config_expands_signal_without_using_last_13_scenarios():
    assert THRESHOLDS == {
        "min_events": 60,
        "min_reverse_events": 8,
        "min_reverse_tasks": 5,
        "max_task_event_share": 0.10,
        "max_events_per_task": 4,
    }
    assert CONFIG["presentations_per_epoch"] == 60
    assert CONFIG["epochs"] == 3
    assert CONFIG["reference_anchor_coef"] == 0.25
    assert CONFIG["sampling"] == "identical_multi_prefix_class_balanced"
    assert DEVELOPMENT_THRESHOLDS["min_class_conditional_shift_gap"] == 0.02
    assert CONFIRMATION_THRESHOLDS["min_class_conditional_shift_gap"] == 0.02


def _rows(method: str) -> list[dict]:
    rows = []
    for index in range(20):
        reverse = index < 3
        selected = "a" if method == "v43" and reverse else "b"
        rows.append(
            {
                "event_id": f"e{index}",
                "selected": selected,
                "decision": (
                    "reverse_preference" if reverse else "rescue_preference"
                ),
                "replay_valid": True,
                "causal_correct": not reverse or method == "v43",
                "selected_terminal_similarity": 0.5,
                "selected_progress_auc": 0.5,
                "margin_b_over_a": (
                    -0.1 if method == "v43" and reverse else 0.2
                ),
            }
        )
    return rows


def _eval(method: str, rows: list[dict]) -> dict:
    return {
        **summarize_evaluation_rows(rows),
        "method": method,
        "evaluation_role": "confirmation",
        "event_set_hash": "events",
        "worker_receives_public_prompt_and_candidates_only": True,
        "offline_outcomes_joined_after_scoring": True,
    }


def test_v43_gate_requires_flips_and_class_conditional_margin_separation():
    mask_rows = _rows("mask")
    v43_rows = _rows("v43")
    protocol = {
        "train_sha256": "train",
        "train_events": 60,
        "config": CONFIG,
        "expected_presented_event_sequence_sha256": "sequence",
        "expected_presented_source_decisions": {
            "rescue_preference": 90,
            "reverse_preference": 90,
        },
        "expected_presented_decisions": {
            "mask": {"b_over_a": 180},
            "v43": {"a_over_b": 90, "b_over_a": 90},
        },
        "gate_thresholds": {
            "development": DEVELOPMENT_THRESHOLDS,
            "confirmation": CONFIRMATION_THRESHOLDS,
        },
        "scope": "test",
    }
    common_run = {
        "train_file_sha256": "train",
        "presentations_per_epoch": 60,
        "active_event_presentations": 180,
        "presented_event_sequence_sha256": "sequence",
        "presented_source_decisions": {
            "rescue_preference": 90,
            "reverse_preference": 90,
        },
        "absolute_margin_coef": 1.0,
        "target_margin": 0.05,
        "reference_anchor_coef": 0.25,
        "loss_definition": (
            "unit_weight*(dpo_shift+absolute_margin)+reference_anchor"
        ),
    }
    gate = build_gate(
        role="confirmation",
        mask_eval=_eval("mask", mask_rows),
        v43_eval=_eval("v43", v43_rows),
        mask_run={
            **common_run,
            "method": "mask",
            "presented_decisions": {"b_over_a": 180},
        },
        v43_run={
            **common_run,
            "method": "v43",
            "presented_decisions": {"a_over_b": 90, "b_over_a": 90},
        },
        mask_rows=mask_rows,
        v43_rows=v43_rows,
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
    assert gate["v43_wins"] == 3
    assert gate["v43_losses"] == 0
    assert gate["mean_rescue_margin_shift"] == 0.0
    assert gate["mean_reverse_margin_shift"] < 0.0
    assert gate["class_conditional_shift_gap"] >= 0.02


def test_v43_runner_freezes_both_identities_before_mining_and_training():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "cloud"
        / "run_toolsandbox_v43_seed42.sh"
    ).read_text(encoding="utf-8")
    freeze_mining = source.index('"$APP_PY" scripts/freeze_toolsandbox_v4_protocol.py')
    freeze_confirmation = source.index(
        '"$APP_PY" scripts/freeze_toolsandbox_v4_protocol.py', freeze_mining + 1
    )
    mining = source.index('"$APP_PY" scripts/audit_toolsandbox_signal.py')
    freeze_training = source.index(
        '"$MODEL_PY" scripts/freeze_toolsandbox_v43_protocol.py'
    )
    train = source.index(
        'CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" '
        "scripts/train_toolsandbox_v43_preference.py"
    )
    development_gate = source.index("gate_pair development")
    confirmation_audit = source.index(
        '"$APP_PY" scripts/audit_toolsandbox_signal.py', mining + 1
    )
    assert freeze_mining < freeze_confirmation < mining < freeze_training < train
    assert train < development_gate < confirmation_audit
    assert "--scenario-offset 85" in source
    assert "--max-events-per-scenario 4" in source
    assert "--scenario-offset 165" in source
    assert "--max-events-per-scenario 1" in source
    assert "scenario-offset 205" not in source
    assert "--reference-anchor-coef 0.25" in source
