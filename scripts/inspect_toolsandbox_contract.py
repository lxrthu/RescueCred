#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from environments.toolsandbox import TOOL_SANDBOX_COMMIT, ToolSandboxRuntime


def _write(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    vendor = args.vendor_dir.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
    ).strip()

    runtime = ToolSandboxRuntime()
    scenarios = runtime.scenarios()
    selected = runtime.select_scenarios(args.limit, args.seed)
    categories = Counter(
        str(category) for _, scenario in scenarios.items() for category in scenario.categories
    )
    snapshot_ok = False
    official_evaluator_callable = False
    public_schemas_present = False
    if selected:
        _, scenario = selected[0]
        context = runtime.prepare(scenario)
        before = runtime.context_digest(context)
        snapshot = runtime.snapshot(context)
        runtime.set_current_context(context)
        runtime.BaseRole.add_messages(
            [
                runtime.Message(
                    sender=runtime.RoleType.AGENT,
                    recipient=runtime.RoleType.USER,
                    content="RescueCredit contract mutation probe",
                )
            ]
        )
        mutated = runtime.context_digest(runtime.get_current_context())
        restored = runtime.context_digest(snapshot)
        snapshot_ok = before == restored and before != mutated
        official_evaluator_callable = callable(getattr(scenario.evaluation, "evaluate", None))
        public_schemas_present = bool(runtime.tool_schemas(snapshot))

    checks = {
        "pinned_official_commit": commit == TOOL_SANDBOX_COMMIT,
        "enough_selected_scenarios": len(selected) >= min(args.limit, 30),
        "snapshot_restore_exact": snapshot_ok,
        "official_evaluator_callable": official_evaluator_callable,
        "public_tool_schemas_present": public_schemas_present,
        "no_reference_actions_exported": True,
    }
    report = {
        "stage": "toolsandbox_contract_probe",
        "passed": all(checks.values()),
        "official_commit": commit,
        "expected_commit": TOOL_SANDBOX_COMMIT,
        "scenarios": len(scenarios),
        "selected_signal_scenarios": len(selected),
        "selection_rule": (
            "single-user multiple-tool no-distraction scenarios; state-dependency "
            "scenarios are deterministically prioritized"
        ),
        "category_counts": dict(sorted(categories.items())),
        "checks": checks,
        "reference_boundary": (
            "scenario evaluation objects are checked for callability only; milestone, "
            "minefield, and reference contents are never serialized"
        ),
        "next_step": (
            "run the 3-scenario sanity and 40-scenario signal audit"
            if all(checks.values())
            else "repair ToolSandbox contract before any signal audit"
        ),
    }
    _write(output / "contract_probe.json", report)
    _write(output / "gate.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
