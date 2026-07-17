#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-causal-events", type=int, default=5)
    parser.add_argument("--min-accuracy-improvement", type=float, default=0.10)
    args = parser.parse_args()
    mask = json.loads(args.mask.read_text(encoding="utf-8"))
    v2 = json.loads(args.v2.read_text(encoding="utf-8"))
    improvement = v2["causal_accuracy"] - mask["causal_accuracy"]
    checks = {
        "same_validation_split": mask["validation_file_sha256"]
        == v2["validation_file_sha256"],
        "enough_causal_validation_events": v2["causal_events"]
        >= args.min_causal_events,
        "v2_improves_causal_accuracy": improvement
        >= args.min_accuracy_improvement,
        "v2_positive_mean_signed_margin": v2["mean_signed_causal_margin"] > 0,
    }
    gate = {
        "passed": all(checks.values()),
        "stage": "route_a_seed42_offline_preference",
        "checks": checks,
        "mask_causal_accuracy": mask["causal_accuracy"],
        "v2_causal_accuracy": v2["causal_accuracy"],
        "accuracy_improvement": improvement,
        "v2_rescue_accuracy": v2["rescue_accuracy"],
        "v2_reverse_accuracy": v2["reverse_accuracy"],
        "scope": "engineering gate only; does not establish AppWorld task success",
        "next_step": (
            "run paired AppWorld dev task-success evaluation"
            if all(checks.values())
            else "stop before AppWorld dev evaluation and inspect the preference learner"
        ),
    }
    args.output.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
