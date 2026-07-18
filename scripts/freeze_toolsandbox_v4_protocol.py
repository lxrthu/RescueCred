#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from environments.toolsandbox import TOOL_SANDBOX_COMMIT, V4_SCENARIO_POOL_PROFILE
from rescuecredit.toolsandbox_audit import DEFAULT_THRESHOLDS
from rescuecredit.toolsandbox_credit import LEXICOGRAPHIC_COMPONENT_ORDER
from rescuecredit.toolsandbox_protocol import (
    REQUIRED_V4_SOURCE_PATHS,
    current_toolsandbox_runtime_identity,
    sha256_file,
)


def protocol_payload(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    plan = args.plan.resolve()
    if not plan.is_file():
        raise FileNotFoundError(plan)
    source_sha256 = {}
    for relative in REQUIRED_V4_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_sha256[relative] = sha256_file(path)
    stage0 = args.stage0_gate.resolve()
    if not stage0.is_file():
        raise FileNotFoundError(stage0)
    stage0_payload = json.loads(stage0.read_text(encoding="utf-8"))
    if stage0_payload.get("passed") is not True:
        raise ValueError("ToolSandbox Stage-0 gate is not passed")
    if stage0_payload.get("official_commit") != TOOL_SANDBOX_COMMIT:
        raise ValueError("Stage-0 ToolSandbox commit mismatch")

    from environments.toolsandbox import ToolSandboxRuntime

    runtime = ToolSandboxRuntime()
    development = runtime.select_scenarios(
        limit=3, seed=args.seed, offset=0, allow_distraction_tools=True
    )
    fresh = runtime.select_scenarios(
        limit=args.limit,
        seed=args.seed,
        offset=args.scenario_offset,
        allow_distraction_tools=True,
    )
    if len(fresh) < 30:
        raise RuntimeError(
            f"insufficient fresh ToolSandbox scenarios at offset "
            f"{args.scenario_offset}: {len(fresh)} < 30"
        )
    development_hashes = [
        hashlib.sha256(name.encode("utf-8")).hexdigest() for name, _ in development
    ]
    fresh_hashes = [
        hashlib.sha256(name.encode("utf-8")).hexdigest() for name, _ in fresh
    ]
    intersection = sorted(set(development_hashes) & set(fresh_hashes))
    if intersection:
        raise ValueError("development and fresh ToolSandbox scenarios overlap")

    return {
        "status": "frozen_before_v4_outcomes",
        "stage": "toolsandbox_v4_lexicographic_regret",
        "toolsandbox_commit": TOOL_SANDBOX_COMMIT,
        "seed": args.seed,
        "scenario_offset": args.scenario_offset,
        "limit": args.limit,
        "horizon": args.horizon,
        "event_search_steps": args.event_search_steps,
        "credit_mode": "lexicographic_v4",
        "scenario_pool_profile": V4_SCENARIO_POOL_PROFILE,
        "lexicographic_component_order": list(LEXICOGRAPHIC_COMPONENT_ORDER),
        "atol": 1e-12,
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "provider": os.getenv("TOOLSANDBOX_LLM_PROVIDER", "deepseek"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://zhi-api.com/v1"),
        "thinking": os.getenv("DEEPSEEK_THINKING", "disabled"),
        "plan": str(plan),
        "plan_sha256": sha256_file(plan),
        "source_sha256": source_sha256,
        "stage0_gate": str(stage0),
        "stage0_gate_sha256": sha256_file(stage0),
        "stage0_official_commit": stage0_payload["official_commit"],
        "toolsandbox_runtime": current_toolsandbox_runtime_identity(
            TOOL_SANDBOX_COMMIT
        ),
        "scenario_identity": {
            "development_offset": 0,
            "development_hashes": development_hashes,
            "fresh_offset": args.scenario_offset,
            "fresh_hashes": fresh_hashes,
            "fresh_count": len(fresh_hashes),
            "intersection": intersection,
        },
        "reference_boundary": {
            "prefix": "reference-free visible worker actions only",
            "branch_score": "official evaluator after execution only",
            "reference_actions": "never read or exported",
            "secret_exported": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, default=Path("refine-logs/TOOLSANDBOX_V4_PLAN.md")
    )
    parser.add_argument(
        "--stage0-gate", type=Path, default=Path("outputs/toolsandbox_stage0/gate.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-offset", type=int, default=40)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--event-search-steps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace frozen protocol: {output}")
    payload = protocol_payload(args, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
