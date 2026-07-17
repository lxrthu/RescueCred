#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--rescue", type=Path, required=True)
    parser.add_argument("--mask-run-summary", type=Path)
    parser.add_argument("--rescue-run-summary", type=Path)
    parser.add_argument("--min-audited-events", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/pilot/gate.json"))
    args = parser.parse_args()
    mask = json.loads(args.mask.read_text(encoding="utf-8"))
    rescue = json.loads(args.rescue.read_text(encoding="utf-8"))
    improvements = {
        "s_off": rescue["s_off"] - mask["s_off"],
        "first_pass": rescue["first_pass"] - mask["first_pass"],
    }
    if args.min_audited_events < 0:
        raise SystemExit("--min-audited-events must be non-negative")
    if args.min_audited_events and args.rescue_run_summary is None:
        raise SystemExit("--rescue-run-summary is required when --min-audited-events is positive")
    if args.mask_run_summary is not None and args.rescue_run_summary is None:
        raise SystemExit("--rescue-run-summary is required with --mask-run-summary")
    audited_events = None
    audit_gate_pass = True
    if args.rescue_run_summary is not None:
        run_summary = json.loads(args.rescue_run_summary.read_text(encoding="utf-8"))
        audited_events = int(run_summary.get("audit_stats", {}).get("valid_audits", 0))
        audit_gate_pass = audited_events >= args.min_audited_events
    metric_gate_pass = any(value > 0 for value in improvements.values())
    sampling_gate = {"checked": False, "passed": True, "mismatches": {}}
    if args.mask_run_summary is not None:
        mask_run = json.loads(args.mask_run_summary.read_text(encoding="utf-8"))
        rescue_run = json.loads(args.rescue_run_summary.read_text(encoding="utf-8"))
        mask_sampling = mask_run.get("sampling", {})
        rescue_sampling = rescue_run.get("sampling", {})
        comparable = {
            "split_hash": (mask_run.get("split_hash"), rescue_run.get("split_hash")),
            "world_size": (mask_run.get("world_size"), rescue_run.get("world_size")),
            "main_interaction_budget": (
                mask_run.get("main_interaction_budget"),
                rescue_run.get("main_interaction_budget"),
            ),
            "visible_curriculum_fraction": (
                mask_sampling.get("visible_curriculum_fraction"),
                rescue_sampling.get("visible_curriculum_fraction"),
            ),
            "visible_pool_hash": (
                mask_sampling.get("visible_pool_hash"),
                rescue_sampling.get("visible_pool_hash"),
            ),
            "assignment_sequence_hash": (
                mask_sampling.get("assignment_sequence_hash"),
                rescue_sampling.get("assignment_sequence_hash"),
            ),
        }
        mismatches = {
            key: {"mask": values[0], "rescue": values[1]}
            for key, values in comparable.items()
            if values[0] is None or values[0] != values[1]
        }
        sampling_gate = {
            "checked": True,
            "passed": not mismatches,
            "mismatches": mismatches,
        }
    passed = metric_gate_pass and audit_gate_pass and sampling_gate["passed"]
    result = {
        "passed": passed,
        "rule": "expand only if RescueCredit improves S_off or First-pass and meets the audit floor",
        "improvements": improvements,
        "metric_gate_pass": metric_gate_pass,
        "audit_gate": {
            "passed": audit_gate_pass,
            "valid_audits": audited_events,
            "minimum": args.min_audited_events,
        },
        "sampling_gate": sampling_gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
