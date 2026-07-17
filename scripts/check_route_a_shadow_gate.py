#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-dir", type=Path, required=True)
    parser.add_argument("--min-valid", type=int, default=10)
    parser.add_argument("--min-nonzero", type=int, default=3)
    args = parser.parse_args()
    summary = json.loads(
        (args.shadow_dir / "shadow_summary.json").read_text(encoding="utf-8")
    )
    checks = {
        "enough_valid_events": summary["valid_events"] >= args.min_valid,
        "causal_support_present": summary["nonzero_events"] >= args.min_nonzero,
        "private_audit_not_read": summary["offline_audit_private_read"] is False,
        "bank_hash_recorded": len(summary["bank_sha256"]) == 64,
    }
    gate = {
        "passed": all(checks.values()),
        "stage": "route_a_shadow_credit_smoke",
        "checks": checks,
        "valid_events": summary["valid_events"],
        "nonzero_events": summary["nonzero_events"],
        "decisions": summary["decisions"],
        "next_step": (
            "build the same-bank Mask vs V2 seed-42 training pair"
            if all(checks.values())
            else "do not train; inspect Shadow failures or zero causal support"
        ),
    }
    (args.shadow_dir / "shadow_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
