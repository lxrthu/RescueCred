import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from environments.toolsandbox.adapter import (
    ToolSandboxRuntime,
    action_schema_complete,
    canonical_action,
    console_namespace_fingerprint,
    controlled_missing_argument,
    score_decision,
)
from rescuecredit.toolsandbox_audit import build_summary_and_gate
from scripts.audit_toolsandbox_signal import (
    Worker,
    _official_score_readonly,
    _snapshot_audit_exact,
)
from scripts.toolsandbox_azure_worker import _validate


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["phone_number", "content"],
            },
        },
    }
]


def test_v4_tiered_pool_preserves_old_prefix_and_has_fresh_holdout():
    categories = types.SimpleNamespace(
        NO_DISTRACTION_TOOLS="no_distraction",
        STATE_DEPENDENCY="state_dependency",
        MULTIPLE_TOOL_CALL="multiple_tool",
        SINGLE_USER_TURN="single_user",
    )
    runtime = ToolSandboxRuntime.__new__(ToolSandboxRuntime)
    runtime.ScenarioCategories = categories
    scenarios = {}
    common = {categories.MULTIPLE_TOOL_CALL, categories.SINGLE_USER_TURN}
    for index in range(20):
        scenarios[f"nd_state_{index:02d}"] = types.SimpleNamespace(
            categories=common
            | {categories.NO_DISTRACTION_TOOLS, categories.STATE_DEPENDENCY},
            starting_context=object(),
        )
        scenarios[f"nd_other_{index:02d}"] = types.SimpleNamespace(
            categories=common | {categories.NO_DISTRACTION_TOOLS},
            starting_context=object(),
        )
        scenarios[f"d_state_{index:02d}"] = types.SimpleNamespace(
            categories=common | {categories.STATE_DEPENDENCY},
            starting_context=object(),
        )
        scenarios[f"d_other_{index:02d}"] = types.SimpleNamespace(
            categories=common,
            starting_context=object(),
        )
    runtime.scenarios = lambda: scenarios
    runtime._agent_tools = lambda context: {
        "safe": types.SimpleNamespace(__module__="safe.tools")
    }

    old_pool = runtime.select_scenarios(limit=100, seed=42)
    expanded = runtime.select_scenarios(
        limit=100, seed=42, allow_distraction_tools=True
    )
    fresh = runtime.select_scenarios(
        limit=40, seed=42, offset=40, allow_distraction_tools=True
    )

    assert len(old_pool) == 40
    assert len(expanded) == 80
    assert expanded[:40] == old_pool
    assert len(fresh) == 40
    assert not ({name for name, _ in old_pool} & {name for name, _ in fresh})


def test_intermediate_official_score_must_not_mutate_continuation_state():
    class Runtime:
        @staticmethod
        def context_digest(context):
            return str(context["value"])

        @staticmethod
        def official_score(scenario, context):
            context["value"] += 1
            return {"similarity": 0.0}

    with pytest.raises(RuntimeError, match="mutated branch continuation state"):
        _official_score_readonly(Runtime(), object(), {"value": 0})


def test_snapshot_gate_requires_checks_and_zero_mismatches():
    assert _snapshot_audit_exact(3, 0, 6, 0) is True
    assert _snapshot_audit_exact(0, 0, 0, 0) is False
    assert _snapshot_audit_exact(3, 1, 6, 0) is False
    assert _snapshot_audit_exact(3, 0, 6, 1) is False


def test_worker_imports_project_code_from_isolated_cwd(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "toolsandbox_azure_worker.py"
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout


def test_worker_restarts_after_timeout_without_cascading_failure(tmp_path: Path):
    marker = tmp_path / "first_request_seen"
    script = tmp_path / "worker.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "from pathlib import Path",
                f"marker = Path({str(marker)!r})",
                "for line in sys.stdin:",
                "    if not marker.exists():",
                "        marker.write_text('seen', encoding='utf-8')",
                "        time.sleep(2)",
                "    print(json.dumps({'ok': True}), flush=True)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    worker = Worker(
        Path(sys.executable), script, tmp_path / "worker.stderr", timeout_sec=0.2
    )
    try:
        with pytest.raises(TimeoutError, match="exceeded"):
            worker.request({"request": 1})
        assert worker.request({"request": 2}) == {"ok": True}
    finally:
        worker.close()


def test_controlled_corruption_uses_only_public_required_field():
    action_b = {
        "tool": "send_message",
        "arguments": {"content": "hello", "phone_number": "+123"},
    }
    result = controlled_missing_argument(action_b, SCHEMAS)
    assert result is not None
    action_a, removed = result
    assert removed == "content"
    assert action_schema_complete(action_b, SCHEMAS)
    assert not action_schema_complete(action_a, SCHEMAS)
    assert action_b["arguments"]["content"] == "hello"


def test_controlled_corruption_abstains_without_complete_public_action():
    incomplete = {"tool": "send_message", "arguments": {"phone_number": "+123"}}
    assert controlled_missing_argument(incomplete, SCHEMAS) is None
    assert controlled_missing_argument(
        {"tool": "unknown", "arguments": {}}, SCHEMAS
    ) is None


def test_action_is_canonical_and_decision_is_three_way():
    action = canonical_action(
        {"tool": "send_message", "arguments": {"z": 1, "a": "x"}}
    )
    assert list(action["arguments"]) == ["a", "z"]
    assert score_decision(1.0) == "rescue_preference"
    assert score_decision(-1.0) == "reverse_preference"
    assert score_decision(1e-14) == "zero_delta"


def test_console_fingerprint_is_semantic_not_object_identity_based():
    def tool(value):
        return value

    left = types.SimpleNamespace(
        locals={"json": json, "tool": tool, "payload": {"x": [1, "a"]}}
    )
    right = types.SimpleNamespace(
        locals={"payload": {"x": [1, "a"]}, "tool": tool, "json": json}
    )
    assert console_namespace_fingerprint(left) == console_namespace_fingerprint(right)
    right.locals["payload"]["x"].append(2)
    assert console_namespace_fingerprint(left) != console_namespace_fingerprint(right)


def test_worker_validation_rejects_unknown_tools_and_allows_repair_abstention():
    result, error = _validate(
        {"tool": "send_message", "arguments": {"content": "x"}},
        SCHEMAS,
        False,
    )
    assert error is None
    assert result["action"]["tool"] == "send_message"
    assert _validate({"tool": "hidden", "arguments": {}}, SCHEMAS, False)[1] == "unknown_tool"
    assert _validate({"abstain": True}, SCHEMAS, True)[0]["abstained"] is True


def _rows(controlled_nonzero=8, natural=3):
    rows = []
    for index in range(20):
        delta = 0.2 if index < controlled_nonzero else 0.0
        rows.append(
            {
                "mode": "controlled_missing_argument",
                "replay_valid": True,
                "decision": score_decision(delta),
                "delta": delta,
                "terminal_delta": delta,
                "decision_basis": "final_official_similarity",
            }
        )
    for index in range(natural):
        rows.append(
            {
                "mode": "natural_visible_error_repair",
                "replay_valid": True,
                "decision": "rescue_preference",
                "delta": 0.1,
                "terminal_delta": 0.1,
                "decision_basis": "final_official_similarity",
            }
        )
    return rows


def test_signal_gate_passes_dense_controlled_and_natural_audit():
    summary, gate = build_summary_and_gate(
        _rows(),
        scenarios_requested=40,
        scenarios_selected=40,
        worker_failures=0,
        snapshot_restore_exact=True,
        official_evaluator_used=True,
    )
    assert gate["passed"] is True
    assert summary["controlled"]["nonzero_rate"] == 0.4
    assert summary["controlled"]["terminal_nonzero_rate"] == 0.4
    assert summary["reference_boundary"]["reference_actions"] == "never read or exported"
    assert "reference-free" in summary["reference_boundary"]["treatment_search_prefix"]


def test_signal_gate_rejects_sparse_or_reference_invalid_run():
    _, gate = build_summary_and_gate(
        _rows(controlled_nonzero=2, natural=0),
        scenarios_requested=40,
        scenarios_selected=40,
        worker_failures=0,
        snapshot_restore_exact=False,
        official_evaluator_used=True,
    )
    assert gate["passed"] is False
    assert gate["checks"]["controlled_signal_density"] is False
    assert gate["checks"]["snapshot_restore_exact"] is False
    assert gate["checks"]["natural_harness_has_coverage"] is False


def test_mechanism_gate_is_reported_separately_from_deployable_harness():
    summary_rows = _rows(controlled_nonzero=8, natural=0)
    _, gate = build_summary_and_gate(
        summary_rows,
        scenarios_requested=40,
        scenarios_selected=40,
        worker_failures=0,
        snapshot_restore_exact=True,
        official_evaluator_used=True,
    )
    assert gate["mechanism_passed"] is True
    assert gate["deployable_harness_passed"] is False
    assert gate["passed"] is False


def test_v4_gate_cannot_pass_without_validated_protocol_lock():
    _, gate = build_summary_and_gate(
        _rows(),
        scenarios_requested=40,
        scenarios_selected=40,
        worker_failures=0,
        snapshot_restore_exact=True,
        official_evaluator_used=True,
        protocol_required=True,
        protocol_validated=False,
    )
    assert gate["mechanism_passed"] is False
    assert gate["deployable_harness_passed"] is False
    assert gate["checks"]["protocol_validated"] is False


def test_natural_coverage_requires_replay_valid_pairs():
    rows = _rows(controlled_nonzero=8, natural=0)
    rows.extend(
        {
            "mode": "natural_visible_error_repair",
            "replay_valid": False,
            "decision": "invalid",
            "delta": None,
        }
        for _ in range(3)
    )
    _, gate = build_summary_and_gate(
        rows,
        scenarios_requested=40,
        scenarios_selected=40,
        worker_failures=0,
        snapshot_restore_exact=True,
        official_evaluator_used=True,
    )
    assert gate["checks"]["natural_harness_has_coverage"] is False


def test_natural_zero_or_reverse_pairs_cannot_authorize_deployable_claim():
    rows = _rows(controlled_nonzero=8, natural=0)
    rows.extend(
        {
            "mode": "natural_visible_error_repair",
            "replay_valid": True,
            "decision": "reverse_preference" if index == 0 else "zero_delta",
            "delta": -0.1 if index == 0 else 0.0,
            "terminal_delta": -0.1 if index == 0 else 0.0,
            "decision_basis": (
                "final_official_similarity" if index == 0 else "all_components_tied"
            ),
        }
        for index in range(3)
    )
    _, gate = build_summary_and_gate(
        rows,
        scenarios_requested=40,
        scenarios_selected=40,
        worker_failures=0,
        snapshot_restore_exact=True,
        official_evaluator_used=True,
    )
    assert gate["mechanism_passed"] is True
    assert gate["deployable_harness_passed"] is False
    assert gate["checks"]["natural_harness_has_nonzero_credit"] is False
    assert gate["checks"]["natural_harness_wins_over_losses"] is False


def test_audit_rows_are_json_serializable():
    json.dumps(_rows(), ensure_ascii=False)
