#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from rescuecredit.appworld_shadow_credit import credit_decision, requirement_progress
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute Route-A credit from existing AppWorld requirement reports"
    )
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--binary-credit-dir", type=Path, required=True)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--bank-offset", type=int, default=20)
    parser.add_argument("--limit", type=int, default=130)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    bank_path = args.bank_dir / "correction_bank.public.jsonl"
    bank = read_jsonl(bank_path)[args.bank_offset : args.bank_offset + args.limit]
    binary_path = args.binary_credit_dir / "shadow_credit.train.jsonl"
    binary = {row["event_id"]: row for row in read_jsonl(binary_path)}
    dense: list[dict] = []
    failures = Counter()

    for index, event in enumerate(bank):
        event_id = event["event_id"]
        if event_id not in binary:
            failures["binary_replay_invalid"] += 1
            continue
        task_id = str(event["task_id"])
        paths = {
            branch: args.experiments_root
            / f"route_a_shadow_{branch}_{args.seed}_{index}"
            / "tasks"
            / task_id
            / "evaluation"
            / "report.md"
            for branch in ("a", "b")
        }
        if not paths["a"].is_file() or not paths["b"].is_file():
            failures["paired_report_missing"] += 1
            continue
        try:
            passed_a, failed_a, score_a = requirement_progress(
                paths["a"].read_text(encoding="utf-8", errors="replace")
            )
            passed_b, failed_b, score_b = requirement_progress(
                paths["b"].read_text(encoding="utf-8", errors="replace")
            )
        except ValueError:
            failures["aggregate_counts_missing"] += 1
            continue
        # A and B must be scored by the same number of official requirements.
        if passed_a + failed_a != passed_b + failed_b:
            failures["requirement_denominator_mismatch"] += 1
            continue
        delta = score_b - score_a
        dense.append(
            {
                "event_id": event_id,
                "return_a": score_a,
                "return_b": score_b,
                "delta": delta,
                "decision": credit_decision(score_a, score_b),
                "passed_a": passed_a,
                "failed_a": failed_a,
                "passed_b": passed_b,
                "failed_b": failed_b,
                "replay_valid": True,
                "reward_source": "appworld_official_requirement_fraction_v1",
                "requirement_text_exported": False,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dense_path = args.output_dir / "dense_shadow_credit.train.jsonl"
    write_jsonl(dense_path, dense)
    decisions = Counter(row["decision"] for row in dense)
    summary = {
        "status": "completed",
        "bank_sha256": file_sha256(bank_path),
        "binary_credit_sha256": file_sha256(binary_path),
        "events_requested": len(bank),
        "valid_events": len(dense),
        "nonzero_events": sum(abs(row["delta"]) > 1e-12 for row in dense),
        "rescue_events": decisions.get("rescue_preference", 0),
        "reverse_events": decisions.get("reverse_preference", 0),
        "zero_events": decisions.get("zero_delta", 0),
        "failure_reasons": dict(failures),
        "reward_source": "AppWorld official evaluator pass/(pass+fail)",
        "requirement_text_read_by_training": False,
        "requirement_text_exported": False,
        "offline_audit_private_read": False,
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "dense_shadow_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
