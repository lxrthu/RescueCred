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


PROTOCOL_STATUS = "frozen_before_v44_candidate_outcomes"
FULL_THRESHOLDS = {
    "min_scenarios": 30,
    "min_valid_pairs": 60,
    "min_nonzero_pairs": 60,
    "min_rescue_pairs": 8,
    "min_reverse_pairs": 8,
    "min_rescue_tasks": 5,
    "min_reverse_tasks": 5,
    "max_pairs_per_task": 4,
    "max_task_pair_share": 0.10,
    "max_worker_failure_rate": 0.10,
}

SOURCE_PATHS = (
    "environments/toolsandbox/__init__.py",
    "environments/toolsandbox/adapter.py",
    "rescuecredit/appworld_shadow_credit.py",
    "rescuecredit/azure_client.py",
    "rescuecredit/toolsandbox_credit.py",
    "rescuecredit/toolsandbox_protocol.py",
    "scripts/toolsandbox_azure_worker.py",
    "scripts/audit_toolsandbox_signal.py",
    "scripts/audit_toolsandbox_v44_candidates.py",
    "scripts/freeze_toolsandbox_v44_candidate_protocol.py",
    "scripts/prepare_toolsandbox_v44_candidate_data.py",
    "scripts/cloud/run_toolsandbox_v44_candidate_audit.sh",
    "scripts/check_llm.py",
    "refine-logs/TOOLSANDBOX_V44_PLAN.md",
    "refine-logs/TOOLSANDBOX_V44_PLAN_20260720.md",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_hashes(selected: list[tuple[str, Any]]) -> list[str]:
    return [
        hashlib.sha256(name.encode("utf-8")).hexdigest() for name, _ in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("sanity", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage0-gate", type=Path, required=True)
    parser.add_argument("--old-training-protocol", type=Path, required=True)
    parser.add_argument("--v43-mining-protocol", type=Path, required=True)
    parser.add_argument("--v43-data-dir", type=Path, required=True)
    parser.add_argument("--development-protocol", type=Path, required=True)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--v42-root", type=Path, required=True)
    parser.add_argument("--v43-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-offset", type=int, default=85)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--event-search-steps", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, required=True)
    parser.add_argument("--max-pairs-per-scenario", type=int, required=True)
    parser.add_argument("--worker-timeout-sec", type=float, default=600.0)
    parser.add_argument("--harness-interface", choices=("tool_id_v2",), default="tool_id_v2")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to replace frozen protocol: {args.output}")
    if args.role == "sanity":
        expected = {"limit": 3, "candidate_count": 2, "max_pairs": 1}
    else:
        expected = {"limit": 40, "candidate_count": 3, "max_pairs": 4}
    actual = {
        "limit": args.limit,
        "candidate_count": args.candidate_count,
        "max_pairs": args.max_pairs_per_scenario,
    }
    if actual != expected:
        raise ValueError(f"V4.4 {args.role} CLI differs from frozen config: {actual}")
    if (
        args.seed != 42
        or args.scenario_offset != 85
        or args.horizon != 8
        or args.event_search_steps != 8
        or args.worker_timeout_sec != 600.0
    ):
        raise ValueError("V4.4 protocol constants changed")

    root = Path(__file__).resolve().parents[1]
    paths = {
        "plan": args.plan.resolve(),
        "stage0": args.stage0_gate.resolve(),
        "old_training": args.old_training_protocol.resolve(),
        "v43_mining": args.v43_mining_protocol.resolve(),
        "v43_manifest": (args.v43_data_dir / "manifest.json").resolve(),
        "v43_gate": (args.v43_data_dir / "data_gate.json").resolve(),
        "development": args.development_protocol.resolve(),
        "confirmation": args.confirmation_protocol.resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen input artifacts: {missing}")
    missing_sources = [relative for relative in SOURCE_PATHS if not (root / relative).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"missing V4.4 source files: {missing_sources}")

    stage0 = _load(paths["stage0"])
    old_training = _load(paths["old_training"])
    v43_mining = _load(paths["v43_mining"])
    v43_manifest = _load(paths["v43_manifest"])
    v43_gate = _load(paths["v43_gate"])
    development = _load(paths["development"])
    confirmation = _load(paths["confirmation"])
    runtime = ToolSandboxRuntime()
    selected = runtime.select_scenarios(
        limit=args.limit,
        seed=args.seed,
        offset=args.scenario_offset,
        allow_distraction_tools=True,
    )
    selected_hashes = _scenario_hashes(selected)
    old_hashes = old_training.get("scenario_identity", {}).get("fresh_hashes", [])
    v43_hashes = v43_mining.get("scenario_identity", {}).get("fresh_hashes", [])
    checks = {
        "stage0_passed_and_pinned": stage0.get("passed") is True
        and stage0.get("official_commit") == TOOL_SANDBOX_COMMIT,
        "exact_training_only_scenarios": len(selected_hashes) == args.limit
        and selected_hashes == old_hashes[: args.limit]
        and selected_hashes == v43_hashes[: args.limit],
        "v43_failure_preserved": v43_gate.get("passed") is False
        and v43_manifest.get("status") == "rejected"
        and v43_manifest.get("events") == 75
        and v43_manifest.get("reverse_tasks") == 4
        and v43_gate.get("reverse_events") == 4,
        "development_and_confirmation_disjoint": not (
            set(selected_hashes)
            & set(development.get("scenario_identity", {}).get("fresh_hashes", []))
        )
        and not (
            set(selected_hashes)
            & set(confirmation.get("scenario_identity", {}).get("fresh_hashes", []))
        ),
        "confirmation_outcomes_absent": not (
            args.v42_root / "fresh_confirm_offset165_h8" / "audit_summary.json"
        ).exists()
        and not (
            args.v43_root / "fresh_confirm_offset165_h8" / "audit_summary.json"
        ).exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V4.4 protocol preflight failed: {checks}")

    payload = {
        "status": PROTOCOL_STATUS,
        "stage": "toolsandbox_v44_reference_free_candidate_diversity",
        "role": args.role,
        "checks": checks,
        "toolsandbox_commit": TOOL_SANDBOX_COMMIT,
        "scenario_pool_profile": V4_SCENARIO_POOL_PROFILE,
        "seed": args.seed,
        "scenario_offset": args.scenario_offset,
        "limit": args.limit,
        "horizon": args.horizon,
        "event_search_steps": args.event_search_steps,
        "candidate_count": args.candidate_count,
        "max_pairs_per_scenario": args.max_pairs_per_scenario,
        "worker_timeout_sec": args.worker_timeout_sec,
        "harness_interface": args.harness_interface,
        "credit_mode": "lexicographic_v4",
        "lexicographic_component_order": list(LEXICOGRAPHIC_COMPONENT_ORDER),
        "atol": 1e-12,
        "thresholds": FULL_THRESHOLDS if args.role == "full" else None,
        "provider": os.getenv("TOOLSANDBOX_LLM_PROVIDER", "deepseek"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://zhi-api.com/v1"),
        "thinking": os.getenv("DEEPSEEK_THINKING", "disabled"),
        "scenario_identity": {
            "fresh_offset": args.scenario_offset,
            "fresh_hashes": selected_hashes,
            "fresh_count": len(selected_hashes),
            "old_training_protocol_sha256": file_sha256(paths["old_training"]),
            "v43_mining_protocol_sha256": file_sha256(paths["v43_mining"]),
        },
        "artifact_identity": {
            name + "_sha256": file_sha256(path) for name, path in paths.items()
        },
        "artifact_paths": {name: str(path) for name, path in paths.items()},
        "source_sha256": {
            relative: file_sha256(root / relative) for relative in SOURCE_PATHS
        },
        "toolsandbox_runtime": current_toolsandbox_runtime_identity(
            TOOL_SANDBOX_COMMIT
        ),
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
