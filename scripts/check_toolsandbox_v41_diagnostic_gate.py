#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


THRESHOLDS = {
    "exact_scenarios": 5,
    "min_schema_complete_scenarios": 4,
    "min_proposal_coverage": 0.8,
    "min_controlled_valid_events": 3,
    "max_worker_failure_rate": 0.1,
}


def build_diagnostic_gate(
    summary: Mapping[str, Any], audit_gate: Mapping[str, Any]
) -> dict[str, Any]:
    proposal = summary.get("proposal_coverage", {})
    controlled = summary.get("controlled", {})
    snapshot = summary.get("snapshot_audit", {})
    audit_checks = audit_gate.get("checks", {})
    checks = {
        "exact_five_unseen_scenarios": (
            summary.get("scenarios_requested") == THRESHOLDS["exact_scenarios"]
            and summary.get("scenarios_selected") == THRESHOLDS["exact_scenarios"]
            and summary.get("scenario_offset") == 80
        ),
        "tool_id_interface_bound": summary.get("harness_interface") == "tool_id_v2",
        "protocol_validated": summary.get("protocol_validated") is True,
        "enough_schema_complete_scenarios": int(
            proposal.get("schema_complete_scenarios", 0)
        )
        >= THRESHOLDS["min_schema_complete_scenarios"],
        "proposal_coverage": float(proposal.get("rate", 0.0))
        >= THRESHOLDS["min_proposal_coverage"],
        "enough_controlled_valid_events": int(controlled.get("valid_events", 0))
        >= THRESHOLDS["min_controlled_valid_events"],
        "worker_failure_rate": float(summary.get("worker_failure_rate", 1.0))
        <= THRESHOLDS["max_worker_failure_rate"],
        "snapshot_restore_exact": snapshot.get("exact") is True,
        "official_evaluator_used": audit_checks.get("official_evaluator_used") is True,
    }
    passed = all(checks.values())
    return {
        "stage": "toolsandbox_v41_tool_id_diagnostic_gate",
        "passed": passed,
        "checks": checks,
        "thresholds": dict(THRESHOLDS),
        "proposal_coverage": proposal,
        "controlled_valid_events": int(controlled.get("valid_events", 0)),
        "worker_failure_rate": float(summary.get("worker_failure_rate", 0.0)),
        "next_step": (
            "run the pre-frozen offset-85 fresh 40-scenario audit"
            if passed
            else "stop before fresh audit and inspect Tool-ID Harness diagnostics"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--audit-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    audit_gate = json.loads(args.audit_gate.read_text(encoding="utf-8"))
    gate = build_diagnostic_gate(summary, audit_gate)
    args.output.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
