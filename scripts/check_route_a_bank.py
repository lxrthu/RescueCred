#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl, validate_public_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--min-events", type=int, default=30)
    args = parser.parse_args()
    manifest = json.loads((args.bank_dir / "manifest.json").read_text(encoding="utf-8"))
    public_path = args.bank_dir / "correction_bank.public.jsonl"
    private_path = args.bank_dir / "offline_audit.private.jsonl"
    public = read_jsonl(public_path)
    private = read_jsonl(private_path)
    for row in public:
        validate_public_record(row)
    public_ids = {row["event_id"] for row in public}
    private_ids = {row["event_id"] for row in private}
    checks = {
        "enough_events": len(public) >= args.min_events,
        "unique_public_ids": len(public_ids) == len(public),
        "private_join_exact": public_ids == private_ids,
        "public_hash_matches": file_sha256(public_path)
        == manifest.get("public_bank_sha256"),
        "private_hash_matches": file_sha256(private_path)
        == manifest.get("private_audit_sha256"),
        "train_only": all(row.get("split") == "train" for row in public),
    }
    passed = all(checks.values())
    gate = {
        "passed": passed,
        "stage": "route_a_frozen_bank",
        "events": len(public),
        "checks": checks,
        "next_step": (
            "run the same-bank Mask vs RescueCredit-v2 seed-42 pilot"
            if passed
            else "fix the bank before any training"
        ),
    }
    (args.bank_dir / "bank_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
