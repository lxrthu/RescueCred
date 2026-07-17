#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_bounded import (
    EXPECTED_EVENTS,
    EXPECTED_EVENT_FILE_SHA256,
    EXPECTED_EVENT_SET_HASH,
    EXPECTED_HORIZONS,
    EXPECTED_SEED,
)
from rescuecredit.route_a_task_eval import event_set_hash


def _ids(path: Path) -> list[str]:
    return [str(row["event_id"]) for row in read_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("--mask-results", type=Path, required=True)
    parser.add_argument("--v2-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = read_jsonl(args.event_file)
    event_ids = [str(row["event_id"]) for row in events]
    mask_ids = _ids(args.mask_results)
    v2_ids = _ids(args.v2_results)
    checks = {
        "exact_event_count": len(event_ids) == EXPECTED_EVENTS,
        "unique_event_ids": len(set(event_ids)) == len(event_ids),
        "exact_event_set_hash": event_set_hash(events) == EXPECTED_EVENT_SET_HASH,
        "exact_event_file_sha256": file_sha256(args.event_file)
        == EXPECTED_EVENT_FILE_SHA256,
        "mask_covers_exact_event_set": set(mask_ids) == set(event_ids)
        and len(mask_ids) == len(event_ids),
        "v2_covers_exact_event_set": set(v2_ids) == set(event_ids)
        and len(v2_ids) == len(event_ids),
        "event_records_are_dev_seed42_fixture": all(
            row.get("split") == "dev" and isinstance(row.get("task_index"), int)
            for row in events
        ),
    }
    if not all(checks.values()):
        raise SystemExit(json.dumps({"passed": False, "checks": checks}, indent=2))
    lock = {
        "status": "frozen_before_bounded_outcomes",
        "stage": "route_a_appworld_bounded_horizon_seed42",
        "seed": EXPECTED_SEED,
        "horizons": list(EXPECTED_HORIZONS),
        "events": EXPECTED_EVENTS,
        "event_set_hash": EXPECTED_EVENT_SET_HASH,
        "event_file_sha256": file_sha256(args.event_file),
        "mask_results_sha256": file_sha256(args.mask_results),
        "v2_results_sha256": file_sha256(args.v2_results),
        "checks": checks,
        "test_split_access": False,
    }
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != lock:
            raise SystemExit("existing bounded protocol lock does not match inputs")
    else:
        write_json(args.output, lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
