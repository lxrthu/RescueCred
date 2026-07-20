import copy
from pathlib import Path

import pytest

from rescuecredit.toolsandbox_active_shadow_v9 import build_two_step_features
from scripts.freeze_toolsandbox_v9_protocol import GATE


def _receipt(tool, content, exception=None):
    return {
        "action": {"tool": tool, "arguments": {"value": content}},
        "content": content,
        "exception": exception,
    }


def _row(second_a="fixed", second_b="done"):
    return {
        "action_a": {"tool": "search", "arguments": {"query": "x"}},
        "action_b": {"tool": "lookup", "arguments": {"query": "x"}},
        "branch_a": {
            "receipts": [
                _receipt("search", "missing id", "ValueError"),
                _receipt("lookup", second_a),
                _receipt("send", "PROHIBITED THIRD RECEIPT"),
            ],
            "score_trace": [{"similarity": 1.0}],
            "ending_context_digest": "protected",
        },
        "branch_b": {
            "receipts": [
                _receipt("lookup", "id 123"),
                _receipt("send", second_b),
                _receipt("delete", "PROHIBITED THIRD RECEIPT"),
            ],
            "score_trace": [{"similarity": -1.0}],
            "ending_context_digest": "protected",
        },
    }


def test_two_step_features_change_with_second_receipt():
    first = build_two_step_features(_row(), hash_dimension=32)
    second = build_two_step_features(_row(second_b="permission denied"), hash_dimension=32)
    assert len(first) == len(second)
    assert first != second


def test_two_step_features_ignore_third_receipt_and_outcomes():
    row = _row()
    original = build_two_step_features(row, hash_dimension=32)
    changed = copy.deepcopy(row)
    changed["branch_a"]["receipts"][2]["content"] = "changed"
    changed["branch_a"]["score_trace"] = [{"similarity": -999.0}]
    changed["branch_a"]["ending_context_digest"] = "changed"
    assert original == build_two_step_features(changed, hash_dimension=32)


def test_two_step_features_support_censored_second_step():
    row = _row()
    row["branch_a"]["receipts"] = row["branch_a"]["receipts"][:1]
    assert build_two_step_features(row, hash_dimension=32)


def test_two_step_features_reject_protected_second_receipt():
    row = _row(second_a='{"official_score": 1.0}')
    with pytest.raises(ValueError, match="protected keys"):
        build_two_step_features(row, hash_dimension=32)


def test_v9_gate_requires_two_step_improvement():
    assert GATE["min_active_cross_task_roc_auc"] == 0.75
    assert GATE["require_two_step_auc_above_one_step"] is True


def test_v9_runner_never_recollects_or_calls_worker():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/cloud/run_toolsandbox_v9_two_step_seed42.sh"
    ).read_text(encoding="utf-8")
    assert "candidate_events.jsonl" in source
    assert "audit_toolsandbox_v44_candidates.py" not in source
    assert "toolsandbox_azure_worker.py" not in source


def test_v9_freeze_recomputes_paired_v7_baseline():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/freeze_toolsandbox_v9_protocol.py"
    ).read_text(encoding="utf-8")
    assert "model/oof_predictions.jsonl" in source
    assert "recomputed_v7_auc" in source
    assert "v7_paired_event_identity" in source
    assert "all(v7_gate[\"integrity_checks\"].values())" in source
    assert 'v7_protocol.get("config") == CONFIG' in source
    assert 'v7_protocol.get("gate") == V7_GATE' in source
