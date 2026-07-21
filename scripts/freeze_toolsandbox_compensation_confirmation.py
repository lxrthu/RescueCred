#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from environments.toolsandbox import (
    TOOL_SANDBOX_COMMIT,
    V4_SCENARIO_POOL_PROFILE,
    ToolSandboxRuntime,
)
from rescuecredit.frozen_bank import file_sha256
from rescuecredit.toolsandbox_credit import LEXICOGRAPHIC_COMPONENT_ORDER
from rescuecredit.toolsandbox_protocol import current_toolsandbox_runtime_identity
from scripts.freeze_toolsandbox_v44_candidate_protocol import PROTOCOL_STATUS


CONFIG = {
    "seed": 42,
    "scenario_offset": 205,
    "limit": 13,
    "horizon": 8,
    "event_search_steps": 8,
    "candidate_count": 3,
    "max_pairs_per_scenario": 4,
    "worker_timeout_sec": 600.0,
}
THRESHOLDS = {
    "min_valid_nonzero_events": 20,
    "min_events_per_direction": 3,
    "min_tasks_per_direction": 3,
    "min_task_bootstrap_direction_probability": 0.95,
    "task_bootstrap_replicates": 20_000,
    "max_invalid_pair_rate": 0.25,
    "max_worker_failure_rate": 0.10,
}
SOURCE_PATHS = (
    "environments/toolsandbox/__init__.py",
    "environments/toolsandbox/adapter.py",
    "rescuecredit/appworld_shadow_credit.py",
    "rescuecredit/azure_client.py",
    "rescuecredit/compensation_trap.py",
    "rescuecredit/frozen_bank.py",
    "rescuecredit/logging.py",
    "rescuecredit/toolsandbox_credit.py",
    "rescuecredit/toolsandbox_protocol.py",
    "scripts/toolsandbox_azure_worker.py",
    "scripts/audit_toolsandbox_signal.py",
    "scripts/audit_toolsandbox_v44_candidates.py",
    "scripts/freeze_toolsandbox_v44_candidate_protocol.py",
    "scripts/freeze_toolsandbox_compensation_confirmation.py",
    "scripts/check_toolsandbox_compensation_confirmation.py",
    "scripts/cloud/run_toolsandbox_compensation_confirmation_seed42.sh",
    "refine-logs/COMPENSATION_TRAP_EXPERIMENT_PLAN_20260721_230022.md",
)


def _scenario_hashes(payload: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {
                "fresh_hashes",
                "scenario_hashes",
                "selected_hashes",
                "selected_scenario_hashes",
                "development_hashes",
                "confirmation_hashes",
            } and isinstance(value, list):
                found.update(str(item) for item in value)
            else:
                found.update(_scenario_hashes(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_scenario_hashes(value))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-gate", type=Path, required=True)
    parser.add_argument("--historical-protocol", type=Path, action="append", required=True)
    parser.add_argument("--historical-output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace frozen confirmation protocol")
    stage0 = json.loads(args.stage0_gate.read_text(encoding="utf-8"))
    if stage0.get("passed") is not True or stage0.get("official_commit") != TOOL_SANDBOX_COMMIT:
        raise ValueError("ToolSandbox stage-0 gate is not valid")
    runtime = ToolSandboxRuntime()
    selected_with_sentinel = runtime.select_scenarios(
        limit=CONFIG["limit"] + 1,
        seed=CONFIG["seed"],
        offset=CONFIG["scenario_offset"],
        allow_distraction_tools=True,
    )
    if len(selected_with_sentinel) != CONFIG["limit"]:
        raise ValueError(
            f"untouched ToolSandbox tail has {len(selected_with_sentinel)} scenarios, expected exactly {CONFIG['limit']}"
        )
    selected = selected_with_sentinel
    selected_hashes = [
        hashlib.sha256(name.encode("utf-8")).hexdigest() for name, _ in selected
    ]
    historical_hashes: set[str] = set()
    historical_identity = {}
    for path in args.historical_protocol:
        payload = json.loads(path.read_text(encoding="utf-8"))
        historical_hashes.update(_scenario_hashes(payload))
        historical_identity[str(path.resolve())] = file_sha256(path)
    overlap = sorted(set(selected_hashes) & historical_hashes)
    if overlap:
        raise RuntimeError({"fresh_scenario_overlap": overlap})
    preexisting_tail_paths = sorted(
        str(path)
        for path in args.historical_output_root.rglob("*offset205*")
        if path.exists()
    )
    if preexisting_tail_paths:
        raise RuntimeError({"preexisting_offset205_artifacts": preexisting_tail_paths})
    root = Path(__file__).resolve().parents[1]
    missing = [path for path in SOURCE_PATHS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError({"missing_confirmation_sources": missing})
    payload = {
        "status": PROTOCOL_STATUS,
        "stage": "toolsandbox_compensation_trap_fresh_confirmation",
        "role": "full",
        "evaluation_role": "untouched_confirmation",
        "toolsandbox_commit": TOOL_SANDBOX_COMMIT,
        "scenario_pool_profile": V4_SCENARIO_POOL_PROFILE,
        **CONFIG,
        "harness_interface": "tool_id_v2",
        "credit_mode": "lexicographic_v4",
        "lexicographic_component_order": list(LEXICOGRAPHIC_COMPONENT_ORDER),
        "atol": 1e-12,
        "thresholds": THRESHOLDS,
        "provider": os.getenv("TOOLSANDBOX_LLM_PROVIDER", "deepseek"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://zhi-api.com/v1"),
        "thinking": os.getenv("DEEPSEEK_THINKING", "disabled"),
        "scenario_identity": {
            "fresh_offset": CONFIG["scenario_offset"],
            "fresh_count": len(selected_hashes),
            "fresh_hashes": selected_hashes,
            "historical_hash_count": len(historical_hashes),
            "historical_overlap": overlap,
        },
        "artifact_paths": {
            "plan": str(args.plan.resolve()),
            "stage0": str(args.stage0_gate.resolve()),
            **{
                f"historical_{index}": str(path.resolve())
                for index, path in enumerate(args.historical_protocol)
            },
        },
        "artifact_identity": {
            "plan_sha256": file_sha256(args.plan),
            "stage0_sha256": file_sha256(args.stage0_gate),
            **{
                f"historical_{index}_sha256": file_sha256(path)
                for index, path in enumerate(args.historical_protocol)
            },
        },
        "historical_protocol_sha256": historical_identity,
        "historical_inventory_files": len(args.historical_protocol),
        "preexisting_offset205_artifacts": [],
        "source_sha256": {
            path: file_sha256(root / path) for path in SOURCE_PATHS
        },
        "toolsandbox_runtime": current_toolsandbox_runtime_identity(TOOL_SANDBOX_COMMIT),
        "labels_inspected_before_freeze": False,
        "reference_boundary": {
            "candidate_inputs": [
                "visible task messages",
                "public Tool-ID schemas",
                "visible receipts",
                "reference-free proposal A",
            ],
            "candidate_outputs": "unranked schema-valid alternatives only",
            "official_outcomes": "joined only after candidates are fixed",
            "reference_actions": "never read or exported",
            "secret_exported": False,
        },
        "claim_boundary": "fixed 13-scenario untouched-tail confirmation of the Compensation Trap; no model selection or adaptive enlargement",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
