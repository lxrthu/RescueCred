#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    commits = {}
    draws = set()
    for line_number, line in enumerate(args.ledger.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        event_id = row["event_id"]
        if row["kind"] == "probability_commit":
            if event_id in commits:
                raise SystemExit(f"duplicate commit at line {line_number}")
            commits[event_id] = row
        elif row["kind"] == "audit_draw":
            if event_id not in commits:
                raise SystemExit(f"draw before commit at line {line_number}")
            if event_id in draws or row["drawn_at_ns"] <= commits[event_id]["committed_at_ns"]:
                raise SystemExit(f"invalid draw order at line {line_number}")
            if row["commit_digest"] != commits[event_id]["digest"]:
                raise SystemExit(f"commit digest mismatch at line {line_number}")
            draws.add(event_id)
    print(json.dumps({"status": "PASS", "commits": len(commits), "draws": len(draws)}))


if __name__ == "__main__":
    main()

