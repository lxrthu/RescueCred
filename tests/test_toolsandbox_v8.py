from pathlib import Path

from rescuecredit.toolsandbox_active_shadow_v8 import build_v8_features


def _summary(content, exception=None, after_tool="send"):
    schema = {
        "type": "function",
        "function": {
            "name": "send",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    after = {**schema, "function": {**schema["function"], "name": after_tool}}
    return {
        "receipt": {
            "action": {"tool": "send", "arguments": {}},
            "content": content,
            "exception": exception,
        },
        "appended_visible_history": [
            {
                "sender": "agent",
                "recipient": "execution",
                "content": "send()",
                "tool_call_exception": None,
            },
            {
                "sender": "execution",
                "recipient": "agent",
                "content": content,
                "tool_call_exception": exception,
            },
        ],
        "schemas_before": [schema],
        "schemas_after": [after],
    }


def _row(summary_a=None, summary_b=None):
    return {
        "action_a": {"tool": "send", "arguments": {"content": "x"}},
        "action_b": {
            "tool": "send",
            "arguments": {"content": "x", "recipient": "+123"},
        },
        "state_summary_a": summary_a or _summary("missing recipient", "ValueError"),
        "state_summary_b": summary_b or _summary('{"status":"ok"}'),
    }


def test_v8_features_change_with_visible_schema_transition():
    unchanged = build_v8_features(_row(), hash_dimension=32)
    changed = build_v8_features(
        _row(summary_b=_summary('{"status":"ok"}', after_tool="confirm")),
        hash_dimension=32,
    )
    assert len(unchanged) == len(changed)
    assert unchanged != changed


def test_v8_features_include_explicit_a_b_state_delta():
    features = build_v8_features(_row(), hash_dimension=32)
    assert len(features) == 16 + 32 + 5 + 5 + 5 + 32


def test_v8_collector_never_calls_official_evaluator():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/collect_toolsandbox_v8_visible_state.py"
    ).read_text(encoding="utf-8")
    assert "official_score(" not in source
    assert '"official_evaluator_called": False' in source
    assert "runtime.snapshot(prefix)" in source


def test_v8_runner_is_one_step_and_reuses_nested_crossfit():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/cloud/run_toolsandbox_v8_visible_state_seed42.sh").read_text(
        encoding="utf-8"
    )
    trainer = (root / "scripts/train_toolsandbox_v8_active_shadow.py").read_text(
        encoding="utf-8"
    )
    assert "collect_toolsandbox_v8_visible_state.py" in runner
    assert "audit_toolsandbox_v44_candidates.py" not in runner
    assert "train_toolsandbox_v7_active_shadow" in trainer
    assert "deployment_ready" not in trainer  # inherited unchanged from reviewed V7


def test_v8_freezes_and_rechecks_worker_identity():
    root = Path(__file__).resolve().parents[1]
    freeze = (root / "scripts/freeze_toolsandbox_v8_protocol.py").read_text(
        encoding="utf-8"
    )
    collector = (root / "scripts/collect_toolsandbox_v8_visible_state.py").read_text(
        encoding="utf-8"
    )
    for field in ("provider", "model", "base_url", "thinking"):
        assert f'"{field}"' in freeze
        assert f'"{field}"' in collector
    assert "worker_environment_matches_v44" in freeze
    assert "worker environment drifted" in collector
