import argparse
import hashlib
import json
from pathlib import Path

import pytest

from environments.toolsandbox import V4_SCENARIO_POOL_PROFILE
from rescuecredit.toolsandbox_audit import DEFAULT_THRESHOLDS
from rescuecredit.toolsandbox_credit import LEXICOGRAPHIC_COMPONENT_ORDER
from rescuecredit.toolsandbox_protocol import REQUIRED_V4_SOURCE_PATHS
from scripts import audit_toolsandbox_signal
from scripts.audit_toolsandbox_signal import _validate_protocol_lock


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_protocol_lock_binds_config_and_sources(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    worker = root / "scripts" / "toolsandbox_azure_worker.py"
    plan = tmp_path / "plan.md"
    plan.write_text("frozen plan", encoding="utf-8")
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
    runtime_identity = {
        "python": "test-python",
        "python_sha256": "python-sha",
        "package_entry": "tool-sandbox-entry",
        "package_entry_sha256": "package-sha",
    }
    lock = {
        "status": "frozen_before_v4_outcomes",
        "toolsandbox_commit": "165848b9a78cead7ca7fe7c89c688b58e6501219",
        "seed": 42,
        "scenario_offset": 40,
        "limit": 40,
        "horizon": 8,
        "event_search_steps": 8,
        "worker_timeout_sec": 600.0,
        "harness_interface": "tool_name_v1",
        "credit_mode": "lexicographic_v4",
        "scenario_pool_profile": V4_SCENARIO_POOL_PROFILE,
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "base_url": "https://zhi-api.com/v1",
        "thinking": "disabled",
        "lexicographic_component_order": list(LEXICOGRAPHIC_COMPONENT_ORDER),
        "atol": 1e-12,
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "plan": str(plan),
        "plan_sha256": _sha(plan),
        "source_sha256": {
            relative: _sha(root / relative) for relative in REQUIRED_V4_SOURCE_PATHS
        },
        "stage0_gate": str(stage0),
        "stage0_gate_sha256": _sha(stage0),
        "toolsandbox_runtime": runtime_identity,
        "scenario_identity": {
            "development_hashes": ["development"],
            "fresh_hashes": ["fresh"],
            "intersection": [],
            "fresh_vs_excluded_intersection": [],
        },
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    args = argparse.Namespace(
        seed=42,
        scenario_offset=40,
        limit=40,
        horizon=8,
        event_search_steps=8,
        worker_timeout_sec=600.0,
        harness_interface="tool_name_v1",
        credit_mode="lexicographic_v4",
    )
    monkeypatch.setenv("TOOLSANDBOX_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://zhi-api.com/v1")
    monkeypatch.setenv("DEEPSEEK_THINKING", "disabled")
    monkeypatch.setattr(
        audit_toolsandbox_signal,
        "current_toolsandbox_runtime_identity",
        lambda expected_commit: runtime_identity,
    )
    lock_sha, validated = _validate_protocol_lock(path, args, worker)
    assert lock_sha == _sha(path)
    assert validated["source_sha256"] == lock["source_sha256"]

    lock["worker_timeout_sec"] = 601.0
    path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="worker_timeout_sec"):
        _validate_protocol_lock(path, args, worker)
    lock["worker_timeout_sec"] = 600.0

    del lock["source_sha256"]["rescuecredit/azure_client.py"]
    path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        _validate_protocol_lock(path, args, worker)


def test_v4_protocol_rejects_scenario_pool_profile_drift(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    worker = root / "scripts" / "toolsandbox_azure_worker.py"
    plan = tmp_path / "plan.md"
    plan.write_text("frozen plan", encoding="utf-8")
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
    runtime_identity = {"identity": "test"}
    lock = {
        "status": "frozen_before_v4_outcomes",
        "toolsandbox_commit": "165848b9a78cead7ca7fe7c89c688b58e6501219",
        "seed": 42,
        "scenario_offset": 40,
        "limit": 40,
        "horizon": 8,
        "event_search_steps": 8,
        "worker_timeout_sec": 600.0,
        "harness_interface": "tool_name_v1",
        "credit_mode": "lexicographic_v4",
        "scenario_pool_profile": "wrong_profile",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "base_url": "https://zhi-api.com/v1",
        "thinking": "disabled",
        "lexicographic_component_order": list(LEXICOGRAPHIC_COMPONENT_ORDER),
        "atol": 1e-12,
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "plan": str(plan),
        "plan_sha256": _sha(plan),
        "source_sha256": {
            relative: _sha(root / relative) for relative in REQUIRED_V4_SOURCE_PATHS
        },
        "stage0_gate": str(stage0),
        "stage0_gate_sha256": _sha(stage0),
        "toolsandbox_runtime": runtime_identity,
        "scenario_identity": {
            "development_hashes": ["development"],
            "fresh_hashes": ["fresh"],
            "intersection": [],
            "fresh_vs_excluded_intersection": [],
        },
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    args = argparse.Namespace(
        seed=42,
        scenario_offset=40,
        limit=40,
        horizon=8,
        event_search_steps=8,
        worker_timeout_sec=600.0,
        harness_interface="tool_name_v1",
        credit_mode="lexicographic_v4",
    )
    monkeypatch.setenv("TOOLSANDBOX_LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(
        audit_toolsandbox_signal,
        "current_toolsandbox_runtime_identity",
        lambda expected_commit: runtime_identity,
    )
    with pytest.raises(ValueError, match="scenario_pool_profile"):
        _validate_protocol_lock(path, args, worker)
