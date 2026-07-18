import argparse
import hashlib
import json
from pathlib import Path

from scripts import freeze_toolsandbox_v4_protocol
from scripts.check_toolsandbox_v41_diagnostic_gate import build_diagnostic_gate


def _diagnostic_summary(coverage=0.8, complete=4, valid=3, failures=0.0):
    return {
        "scenarios_requested": 5,
        "scenarios_selected": 5,
        "scenario_offset": 80,
        "harness_interface": "tool_id_v2",
        "protocol_validated": True,
        "proposal_coverage": {
            "schema_complete_scenarios": complete,
            "selected_scenarios": 5,
            "rate": coverage,
            "error_types": {},
        },
        "controlled": {"valid_events": valid},
        "worker_failure_rate": failures,
        "snapshot_audit": {"exact": True},
    }


def test_v41_diagnostic_gate_passes_only_with_legal_coverage_and_valid_pairs():
    gate = build_diagnostic_gate(
        _diagnostic_summary(), {"checks": {"official_evaluator_used": True}}
    )
    assert gate["passed"] is True
    failed = build_diagnostic_gate(
        _diagnostic_summary(coverage=0.6, complete=3, valid=2, failures=0.2),
        {"checks": {"official_evaluator_used": True}},
    )
    assert failed["passed"] is False
    assert failed["checks"]["proposal_coverage"] is False
    assert failed["checks"]["enough_controlled_valid_events"] is False


def test_v41_protocol_excludes_all_previously_observed_scenarios(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    plan = tmp_path / "plan.md"
    plan.write_text("v4.1", encoding="utf-8")
    stage0 = tmp_path / "stage0.json"
    stage0.write_text(
        json.dumps(
            {
                "passed": True,
                "official_commit": "165848b9a78cead7ca7fe7c89c688b58e6501219",
            }
        ),
        encoding="utf-8",
    )

    def scenario_hash(index):
        return hashlib.sha256(f"scenario_{index}".encode()).hexdigest()

    old = tmp_path / "old_protocol.json"
    old.write_text(
        json.dumps(
            {
                "scenario_identity": {
                    "development_hashes": [scenario_hash(index) for index in range(3)],
                    "fresh_hashes": [scenario_hash(index) for index in range(40, 80)],
                }
            }
        ),
        encoding="utf-8",
    )

    class Runtime:
        @staticmethod
        def select_scenarios(limit, seed, offset, allow_distraction_tools):
            return [
                (f"scenario_{index}", object())
                for index in range(offset, offset + limit)
            ]

    import environments.toolsandbox as toolsandbox

    monkeypatch.setattr(toolsandbox, "ToolSandboxRuntime", Runtime)
    monkeypatch.setattr(
        freeze_toolsandbox_v4_protocol,
        "current_toolsandbox_runtime_identity",
        lambda commit: {"runtime": "test"},
    )
    args = argparse.Namespace(
        plan=plan,
        stage0_gate=stage0,
        seed=42,
        scenario_offset=80,
        limit=5,
        minimum_scenarios=5,
        horizon=4,
        event_search_steps=4,
        worker_timeout_sec=600.0,
        harness_interface="tool_id_v2",
        exclude_protocol=[old],
    )
    payload = freeze_toolsandbox_v4_protocol.protocol_payload(args, root)
    identity = payload["scenario_identity"]
    assert payload["harness_interface"] == "tool_id_v2"
    assert identity["fresh_count"] == 5
    assert identity["fresh_vs_excluded_intersection"] == []
    assert len(identity["excluded_protocols"]) == 1


def test_v41_runner_freezes_both_sets_and_gates_before_fresh_audit():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "cloud" / "run_toolsandbox_v41_toolid_audit.sh"
    ).read_text(encoding="utf-8")
    diagnostic_freeze = source.index('--output "$DIAGNOSTIC_LOCK"')
    fresh_freeze = source.index('--output "$FRESH_LOCK"')
    audit_command = '"$APP_PY" scripts/audit_toolsandbox_signal.py'
    first_audit = source.index(audit_command)
    diagnostic_gate = source.index(
        '"$MODEL_PY" scripts/check_toolsandbox_v41_diagnostic_gate.py'
    )
    second_audit = source.index(audit_command, first_audit + 1)
    assert diagnostic_freeze < first_audit
    assert fresh_freeze < first_audit
    assert first_audit < diagnostic_gate < second_audit
    assert '--exclude-protocol "$OLD_LOCK"' in source
    assert '--exclude-protocol "$DIAGNOSTIC_LOCK"' in source
