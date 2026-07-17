#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--shadow-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-bank-events", type=int, default=300)
    parser.add_argument("--min-nonzero-events", type=int, default=100)
    parser.add_argument("--min-nonzero-tasks", type=int, default=30)
    parser.add_argument("--min-rescue", type=int, default=25)
    parser.add_argument("--min-reverse", type=int, default=25)
    parser.add_argument("--min-validation-nonzero", type=int, default=15)
    parser.add_argument("--min-replay-valid-rate", type=float, default=0.9)
    parser.add_argument("--max-task-nonzero-share", type=float, default=0.1)
    args = parser.parse_args()

    bank = json.loads((args.bank_dir / "manifest.json").read_text(encoding="utf-8"))
    shadow = json.loads(
        (args.shadow_dir / "shadow_summary.json").read_text(encoding="utf-8")
    )
    data = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    all_stats = data["all"]
    validation = data["validation"]
    decisions = all_stats["decisions"]
    checks = {
        "train_only_bank": bank.get("split") == "train"
        and bank.get("hard_boundary", {}).get("dev_or_test_events") == 0,
        "enough_bank_events": int(bank.get("events", 0)) >= args.min_bank_events,
        "enough_tasks_with_events": int(bank.get("tasks_with_events", 0)) >= 30,
        "replay_valid_rate": float(shadow.get("replay_valid_rate", 0.0))
        >= args.min_replay_valid_rate,
        "cache_has_no_conflicts": int(shadow.get("cache_conflicts", -1)) == 0,
        "worker_isolated": shadow.get("worker_cwd_isolated") is True
        and shadow.get("worker_benchmark_root_in_environment") is False,
        "enough_nonzero_events": int(all_stats.get("nonzero_events", 0))
        >= args.min_nonzero_events,
        "enough_nonzero_tasks": int(all_stats.get("nonzero_tasks", 0))
        >= args.min_nonzero_tasks,
        "nonzero_signal_not_task_dominated": float(
            all_stats.get("max_task_nonzero_share", 1.0)
        )
        <= args.max_task_nonzero_share,
        "enough_rescue_events": int(decisions.get("rescue_preference", 0))
        >= args.min_rescue,
        "enough_reverse_events": int(decisions.get("reverse_preference", 0))
        >= args.min_reverse,
        "validation_has_causal_signal": int(validation.get("nonzero_events", 0))
        >= args.min_validation_nonzero,
        "task_disjoint_split": int(data.get("train_validation_task_overlap", -1))
        == 0,
        "no_private_audit_training": data.get("private_audit_read") is False,
    }
    result = {
        "passed": all(checks.values()),
        "stage": "route_a_v3_expanded_causal_data",
        "checks": checks,
        "bank_events": bank.get("events"),
        "tasks_with_events": bank.get("tasks_with_events"),
        "replay_valid_rate": shadow.get("replay_valid_rate"),
        "all": all_stats,
        "train": data["train"],
        "validation": validation,
        "thresholds": {
            "min_bank_events": args.min_bank_events,
            "min_nonzero_events": args.min_nonzero_events,
            "min_nonzero_tasks": args.min_nonzero_tasks,
            "min_rescue": args.min_rescue,
            "min_reverse": args.min_reverse,
            "min_validation_nonzero": args.min_validation_nonzero,
            "min_replay_valid_rate": args.min_replay_valid_rate,
            "max_task_nonzero_share": args.max_task_nonzero_share,
        },
        "next_step": (
            "freeze a matched Mask vs V3 training pilot"
            if all(checks.values())
            else "do not train; expand or repair causal data coverage"
        ),
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
