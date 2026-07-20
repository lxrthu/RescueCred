#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from environments.toolsandbox import TOOL_SANDBOX_COMMIT, V4_SCENARIO_POOL_PROFILE, ToolSandboxRuntime
from rescuecredit.frozen_bank import file_sha256
from rescuecredit.toolsandbox_credit import LEXICOGRAPHIC_COMPONENT_ORDER
from rescuecredit.toolsandbox_protocol import current_toolsandbox_runtime_identity
from scripts.freeze_toolsandbox_v44_candidate_protocol import FULL_THRESHOLDS, PROTOCOL_STATUS


SOURCE_PATHS = (
    "environments/toolsandbox/__init__.py",
    "environments/toolsandbox/adapter.py",
    "rescuecredit/appworld_shadow_credit.py",
    "rescuecredit/azure_client.py",
    "rescuecredit/frozen_bank.py",
    "rescuecredit/logging.py",
    "rescuecredit/toolsandbox_credit.py",
    "rescuecredit/toolsandbox_protocol.py",
    "scripts/toolsandbox_azure_worker.py",
    "scripts/audit_toolsandbox_signal.py",
    "scripts/audit_toolsandbox_v44_candidates.py",
    "scripts/freeze_toolsandbox_v44_candidate_protocol.py",
    "scripts/freeze_toolsandbox_v45_candidate_protocol.py",
    "scripts/prepare_toolsandbox_v44_candidate_data.py",
    "scripts/cloud/run_toolsandbox_v45_seed42.sh",
    "refine-logs/TOOLSANDBOX_V45_PLAN.md",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hashes(runtime: ToolSandboxRuntime, offset: int) -> list[str]:
    selected = runtime.select_scenarios(limit=40, seed=42, offset=offset, allow_distraction_tools=True)
    return [hashlib.sha256(name.encode("utf-8")).hexdigest() for name, _ in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-role", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage0-gate", type=Path, required=True)
    parser.add_argument("--v44-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-offset", type=int, required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--event-search-steps", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--max-pairs-per-scenario", type=int, default=4)
    parser.add_argument("--worker-timeout-sec", type=float, default=600.0)
    parser.add_argument("--harness-interface", choices=("tool_id_v2",), default="tool_id_v2")
    args = parser.parse_args()

    expected_offset = 125 if args.evaluation_role == "development" else 165
    frozen = (args.seed, args.scenario_offset, args.limit, args.horizon, args.event_search_steps,
              args.candidate_count, args.max_pairs_per_scenario, args.worker_timeout_sec)
    if frozen != (42, expected_offset, 40, 8, 8, 3, 4, 600.0):
        raise ValueError(f"V4.5 {args.evaluation_role} candidate protocol changed: {frozen}")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace frozen protocol: {args.output}")

    root = Path(__file__).resolve().parents[1]
    artifacts = {
        "plan": args.plan.resolve(),
        "stage0": args.stage0_gate.resolve(),
        "v44_protocol": (args.v44_root / "full_protocol_lock.json").resolve(),
        "v44_summary": (args.v44_root / "full_offset85_h8" / "audit_summary.json").resolve(),
        "v44_manifest": (args.v44_root / "data" / "manifest.json").resolve(),
        "v44_data_gate": (args.v44_root / "data" / "data_gate.json").resolve(),
    }
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    missing += [str(root / path) for path in SOURCE_PATHS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing V4.5 protocol inputs: {missing}")
    stage0, v44_protocol = _load(artifacts["stage0"]), _load(artifacts["v44_protocol"])
    v44_summary, v44_manifest = _load(artifacts["v44_summary"]), _load(artifacts["v44_manifest"])
    v44_gate = _load(artifacts["v44_data_gate"])
    runtime = ToolSandboxRuntime()
    train_hashes = v44_protocol.get("scenario_identity", {}).get("fresh_hashes", [])
    development_hashes, confirmation_hashes = _hashes(runtime, 125), _hashes(runtime, 165)
    selected_hashes = development_hashes if args.evaluation_role == "development" else confirmation_hashes
    checks = {
        "stage0_passed_and_pinned": stage0.get("passed") is True and stage0.get("official_commit") == TOOL_SANDBOX_COMMIT,
        "v44_training_gate_passed": v44_gate.get("passed") is True and v44_manifest.get("status") == "frozen" and v44_manifest.get("events") == 126,
        "v44_training_protocol_validated": v44_summary.get("protocol_validated") is True and v44_summary.get("protocol_lock_sha256") == file_sha256(artifacts["v44_protocol"]),
        "exact_evaluation_identity": len(selected_hashes) == 40,
        "train_development_confirmation_disjoint": not (set(train_hashes) & set(development_hashes)) and not (set(train_hashes) & set(confirmation_hashes)) and not (set(development_hashes) & set(confirmation_hashes)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V4.5 candidate preflight failed: {checks}")

    payload = {
        # The V4.4 executor validates this status/role; evaluation_role is an
        # additional immutable V4.5 binding.
        "status": PROTOCOL_STATUS,
        "stage": "toolsandbox_v45_same_distribution_candidate_evaluation",
        "role": "full",
        "evaluation_role": args.evaluation_role,
        "checks": checks,
        "toolsandbox_commit": TOOL_SANDBOX_COMMIT,
        "scenario_pool_profile": V4_SCENARIO_POOL_PROFILE,
        "seed": 42,
        "scenario_offset": expected_offset,
        "limit": 40,
        "horizon": 8,
        "event_search_steps": 8,
        "candidate_count": 3,
        "max_pairs_per_scenario": 4,
        "worker_timeout_sec": 600.0,
        "harness_interface": "tool_id_v2",
        "credit_mode": "lexicographic_v4",
        "lexicographic_component_order": list(LEXICOGRAPHIC_COMPONENT_ORDER),
        "atol": 1e-12,
        "thresholds": FULL_THRESHOLDS,
        "provider": os.getenv("TOOLSANDBOX_LLM_PROVIDER", "deepseek"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://zhi-api.com/v1"),
        "thinking": os.getenv("DEEPSEEK_THINKING", "disabled"),
        "scenario_identity": {"fresh_offset": expected_offset, "fresh_hashes": selected_hashes, "fresh_count": 40},
        "artifact_identity": {name + "_sha256": file_sha256(path) for name, path in artifacts.items()},
        "artifact_paths": {name: str(path) for name, path in artifacts.items()},
        "source_sha256": {path: file_sha256(root / path) for path in SOURCE_PATHS},
        "toolsandbox_runtime": current_toolsandbox_runtime_identity(TOOL_SANDBOX_COMMIT),
        "reference_boundary": {
            "candidate_inputs": ["visible task messages", "public Tool-ID schemas", "visible receipts", "reference-free proposal A"],
            "candidate_outputs": "unranked schema-valid alternatives only",
            "official_outcomes": "joined only after candidates are fixed",
            "reference_actions": "never read or exported",
            "secret_exported": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
