#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.logging import write_json
from rescuecredit.route_a_immediate import immediate_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--min-valid", type=int, default=40)
    parser.add_argument("--min-nonzero", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    gate = immediate_gate(
        summary,
        min_valid=args.min_valid,
        min_nonzero=args.min_nonzero,
    )
    write_json(args.output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
