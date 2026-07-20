#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.deltaguard_protocol import PUBLIC_EVENT_FIELDS, export_public_event
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-events", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("public bank output already exists")
    rows = []
    seen = set()
    excluded = {"not_exact_pair": 0}
    for path in args.raw_events:
        for raw in read_jsonl(path):
            if raw.get("replay_valid") is not True or raw.get("decision") not in {
                "rescue_preference",
                "reverse_preference",
            }:
                excluded["not_exact_pair"] += 1
                continue
            public = export_public_event(raw)
            event_id = str(public["event_id"])
            if event_id in seen:
                raise ValueError(f"duplicate event id across raw banks: {event_id}")
            seen.add(event_id)
            rows.append(public)
    rows.sort(key=lambda row: str(row["event_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    manifest = {
        "status": "completed",
        "stage": "toolsandbox_deltaguard_public_bank_export",
        "events": len(rows),
        "conditioning_scope": "pre-existing exact-Shadow-informative pair bank; source selection and acquisition within this bank are label-blind",
        "eligibility_filter": "replay_valid == true and decision in {rescue_preference, reverse_preference}",
        "excluded": excluded,
        "public_fields": sorted(PUBLIC_EVENT_FIELDS),
        "protected_fields_exported": [],
        "raw_source_sha256": [file_sha256(path) for path in args.raw_events],
        "public_bank_sha256": file_sha256(args.output),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
