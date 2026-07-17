#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_preference import stratified_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    bank_path = args.bank_dir / "correction_bank.public.jsonl"
    credit_path = args.dense_dir / "dense_shadow_credit.train.jsonl"
    bank = {row["event_id"]: row for row in read_jsonl(bank_path)}
    credits = read_jsonl(credit_path)
    rows = []
    for credit in credits:
        event_id = credit["event_id"]
        if event_id not in bank:
            raise ValueError(f"dense credit event absent from frozen bank: {event_id}")
        event = bank[event_id]
        rows.append(
            {
                "event_id": event_id,
                "task_id": event["task_id"],
                "prompt": event["prompt"],
                "action_a": event["action_a"],
                "action_b": event["action_b"],
                "delta": credit["delta"],
                "decision": credit["decision"],
                "reward_source": credit["reward_source"],
            }
        )
    train, validation = stratified_split(rows, args.seed, args.validation_fraction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    manifest = {
        "status": "frozen",
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "bank_sha256": file_sha256(bank_path),
        "dense_credit_sha256": file_sha256(credit_path),
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(validation_path),
        "train_events": len(train),
        "validation_events": len(validation),
        "train_decisions": dict(Counter(row["decision"] for row in train)),
        "validation_decisions": dict(
            Counter(row["decision"] for row in validation)
        ),
        "private_audit_read": False,
        "shared_by": ["mask", "rescuecredit_v2_full_credit"],
        "scope": "offline preference pilot; not AppWorld task-success evidence",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
