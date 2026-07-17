#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from rescuecredit.appworld_shadow_credit import credit_decision
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json


def task_rank(seed: int, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def counts(rows: list[dict]) -> dict:
    decisions = Counter(row["decision"] for row in rows)
    causal = [row for row in rows if row["decision"] != "zero_delta"]
    causal_by_task = Counter(row["task_id"] for row in causal)
    return {
        "events": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        "nonzero_events": len(causal),
        "nonzero_tasks": len({row["task_id"] for row in causal}),
        "max_nonzero_events_per_task": max(causal_by_task.values(), default=0),
        "max_task_nonzero_share": max(causal_by_task.values(), default=0)
        / max(1, len(causal)),
        "decisions": dict(decisions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a task-disjoint V2.1 preference dataset"
    )
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-task-fraction", type=float, default=0.2)
    parser.add_argument("--min-abs-delta", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.validation_task_fraction < 1.0:
        raise ValueError("--validation-task-fraction must be in (0, 1)")
    if args.min_abs_delta <= 0:
        raise ValueError("--min-abs-delta must be positive")

    bank_path = args.bank_dir / "correction_bank.public.jsonl"
    credit_path = args.dense_dir / "dense_shadow_credit.train.jsonl"
    bank = {row["event_id"]: row for row in read_jsonl(bank_path)}
    credits = read_jsonl(credit_path)
    rows = []
    for credit in credits:
        event_id = str(credit["event_id"])
        if event_id not in bank:
            raise ValueError(f"dense credit absent from V2.1 bank: {event_id}")
        event = bank[event_id]
        raw_delta = float(credit["delta"])
        causal_delta = raw_delta if abs(raw_delta) >= args.min_abs_delta else 0.0
        rows.append(
            {
                "event_id": event_id,
                "task_id": str(event["task_id"]),
                "prompt": event["prompt"],
                "action_a": event["action_a"],
                "action_b": event["action_b"],
                "delta": causal_delta,
                "raw_delta": raw_delta,
                "decision": credit_decision(0.0, causal_delta),
                "variant_size": int(event.get("variant_size", 1)),
                "variant_kind": str(event.get("variant_kind", "unknown")),
                "call_index": int(event["call_index"]),
                "missing_parameters": list(event.get("missing_parameters", [])),
                "reward_source": credit["reward_source"],
            }
        )

    tasks = sorted(
        {row["task_id"] for row in rows}, key=lambda task: task_rank(args.seed, task)
    )
    validation_task_count = max(
        1, int(math.ceil(len(tasks) * args.validation_task_fraction))
    )
    validation_tasks = set(tasks[:validation_task_count])
    train = sorted(
        [row for row in rows if row["task_id"] not in validation_tasks],
        key=lambda row: row["event_id"],
    )
    validation = sorted(
        [row for row in rows if row["task_id"] in validation_tasks],
        key=lambda row: row["event_id"],
    )
    if {row["task_id"] for row in train} & {row["task_id"] for row in validation}:
        raise AssertionError("task leakage between V2.1 train and validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    manifest = {
        "status": "frozen",
        "schema_version": "rescuecredit.route_a_v2_1_data.v1",
        "seed": args.seed,
        "validation_task_fraction": args.validation_task_fraction,
        "min_abs_delta": args.min_abs_delta,
        "bank_sha256": file_sha256(bank_path),
        "dense_credit_sha256": file_sha256(credit_path),
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(validation_path),
        "all": counts(rows),
        "train": counts(train),
        "validation": counts(validation),
        "train_validation_task_overlap": 0,
        "private_audit_read": False,
        "shared_by": ["mask_correction", "rescuecredit_v3"],
        "scope": "train-only AppWorld causal preference data; no dev or test access",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
