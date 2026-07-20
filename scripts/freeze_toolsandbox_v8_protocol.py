#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from environments.toolsandbox import TOOL_SANDBOX_COMMIT
from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_protocol import current_toolsandbox_runtime_identity
from scripts.freeze_toolsandbox_v7_protocol import CONFIG, GATE


PROTOCOL_STATUS = "frozen_before_toolsandbox_v8_visible_state_probe"
SOURCE_PATHS = (
    "rescuecredit/toolsandbox_active_shadow.py",
    "rescuecredit/toolsandbox_active_shadow_v8.py",
    "rescuecredit/toolsandbox_router.py",
    "rescuecredit/toolsandbox_selective_router.py",
    "scripts/freeze_toolsandbox_v8_protocol.py",
    "scripts/collect_toolsandbox_v8_visible_state.py",
    "scripts/build_toolsandbox_v8_features.py",
    "scripts/train_toolsandbox_v8_active_shadow.py",
    "scripts/train_toolsandbox_v7_active_shadow.py",
    "scripts/check_toolsandbox_v8_gate.py",
    "scripts/check_toolsandbox_v7_gate.py",
    "scripts/cloud/run_toolsandbox_v8_visible_state_seed42.sh",
    "refine-logs/TOOLSANDBOX_V8_VISIBLE_STATE_PLAN.md",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v44-root", type=Path, required=True)
    parser.add_argument("--v5-root", type=Path, required=True)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    v44_lock_path = args.v44_root / "full_protocol_lock.json"
    raw_path = args.v44_root / "full_offset85_h8/candidate_events.jsonl"
    raw_summary_path = args.v44_root / "full_offset85_h8/audit_summary.json"
    train_path = args.v44_root / "data/train.jsonl"
    train_manifest_path = args.v44_root / "data/manifest.json"
    v5_cache = args.v5_root / "features/train_features.pt"
    v7_gate_path = args.v7_root / "feasibility_gate.json"
    required = [
        v44_lock_path,
        raw_path,
        raw_summary_path,
        train_path,
        train_manifest_path,
        v5_cache,
        v7_gate_path,
        args.worker_script,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    missing += [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    v44_lock = load(v44_lock_path)
    raw_summary = load(raw_summary_path)
    train_manifest = load(train_manifest_path)
    v7_gate = load(v7_gate_path)
    train_rows = read_jsonl(train_path)
    worker_identity = {
        "provider": v44_lock["provider"],
        "model": v44_lock["model"],
        "base_url": v44_lock["base_url"],
        "thinking": v44_lock["thinking"],
    }
    current_worker_identity = {
        "provider": os.getenv("TOOLSANDBOX_LLM_PROVIDER", "deepseek"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://zhi-api.com/v1"),
        "thinking": os.getenv("DEEPSEEK_THINKING", "disabled"),
    }
    checks = {
        "v44_full_protocol_bound": raw_summary.get("protocol_lock_sha256")
        == file_sha256(v44_lock_path),
        "raw_events_bound": raw_summary.get("event_file_sha256")
        == file_sha256(raw_path),
        "train_events_bound": train_manifest.get("source_event_sha256")
        == file_sha256(raw_path)
        and train_manifest.get("train_sha256") == file_sha256(train_path),
        "snapshot_exact": raw_summary.get("snapshot_audit", {}).get("exact") is True,
        "v7_failure_preserved": v7_gate.get("passed") is False
        and float(v7_gate.get("active_cross_task_roc_auc", 1.0)) < 0.75,
        "expected_event_count": len(train_rows) == 126,
        "worker_frozen": file_sha256(args.worker_script)
        == v44_lock.get("source_sha256", {}).get("scripts/toolsandbox_azure_worker.py"),
        "runtime_frozen": current_toolsandbox_runtime_identity(TOOL_SANDBOX_COMMIT)
        == v44_lock.get("toolsandbox_runtime"),
        "worker_environment_matches_v44": current_worker_identity
        == worker_identity,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    replay_config = {
        key: v44_lock[key]
        for key in (
            "seed",
            "scenario_offset",
            "limit",
            "horizon",
            "event_search_steps",
            "candidate_count",
            "max_pairs_per_scenario",
            "worker_timeout_sec",
            "harness_interface",
        )
    }
    protocol = {
        "status": PROTOCOL_STATUS,
        "stage": "toolsandbox_v8_explicit_visible_state_probe_seed42",
        "checks": checks,
        "config": CONFIG,
        "gate": GATE,
        "replay_config": replay_config,
        "scenario_identity": v44_lock["scenario_identity"],
        "toolsandbox_runtime": v44_lock["toolsandbox_runtime"],
        "worker_identity": worker_identity,
        "v44_protocol_sha256": file_sha256(v44_lock_path),
        "raw_events": str(raw_path),
        "raw_events_sha256": file_sha256(raw_path),
        "train_file": str(train_path),
        "train_file_sha256": file_sha256(train_path),
        "v5_feature_cache": str(v5_cache),
        "v5_feature_cache_sha256": file_sha256(v5_cache),
        "v7_gate_sha256": file_sha256(v7_gate_path),
        "worker_script_sha256": file_sha256(args.worker_script),
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "reference_boundary": "one isolated step; agent-visible history and public schemas only; no official evaluator or hidden database/context export",
        "scope": "V8 nested cross-task feasibility using explicit one-step visible state summaries",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
