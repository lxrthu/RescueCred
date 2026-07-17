#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import read_jsonl
from rescuecredit.route_a_task_eval import paired_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mask_summary = json.loads(
        (args.mask_dir / "eval_summary.json").read_text(encoding="utf-8")
    )
    v2_summary = json.loads(
        (args.v2_dir / "eval_summary.json").read_text(encoding="utf-8")
    )
    gate = paired_gate(
        mask_summary,
        v2_summary,
        read_jsonl(args.mask_dir / "task_results.jsonl"),
        read_jsonl(args.v2_dir / "task_results.jsonl"),
        min_events=args.min_events,
    )
    args.output.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
