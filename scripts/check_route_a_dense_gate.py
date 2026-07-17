#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--min-valid", type=int, default=100)
    parser.add_argument("--min-nonzero", type=int, default=15)
    parser.add_argument("--min-rescue", type=int, default=5)
    parser.add_argument("--min-reverse", type=int, default=3)
    args = parser.parse_args()
    summary = json.loads(
        (args.dense_dir / "dense_shadow_summary.json").read_text(encoding="utf-8")
    )
    checks = {
        "enough_valid_events": summary["valid_events"] >= args.min_valid,
        "enough_nonzero_events": summary["nonzero_events"] >= args.min_nonzero,
        "positive_causal_support": summary["rescue_events"] >= args.min_rescue,
        "negative_causal_support": summary["reverse_events"] >= args.min_reverse,
        "no_private_audit": summary["offline_audit_private_read"] is False,
        "no_requirement_text_export": summary["requirement_text_exported"] is False,
    }
    passed = all(checks.values())
    gate = {
        "passed": passed,
        "stage": "route_a_dense_shadow_credit",
        "checks": checks,
        "valid_events": summary["valid_events"],
        "nonzero_events": summary["nonzero_events"],
        "rescue_events": summary["rescue_events"],
        "reverse_events": summary["reverse_events"],
        "next_step": (
            "build the same-bank Mask vs V2 seed-42 training pair"
            if passed
            else "do not train; dense official requirement credit is still insufficient"
        ),
    }
    (args.dense_dir / "dense_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
