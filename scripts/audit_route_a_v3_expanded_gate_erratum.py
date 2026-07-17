#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256


def expected_balanced_counts(
    presentations_per_epoch: int, epochs: int
) -> dict[str, int]:
    counts = {"rescue_preference": 0, "reverse_preference": 0}
    for epoch in range(epochs):
        rescue = presentations_per_epoch // 2
        reverse = presentations_per_epoch // 2
        if presentations_per_epoch % 2:
            if epoch % 2 == 0:
                rescue += 1
            else:
                reverse += 1
        counts["rescue_preference"] += rescue
        counts["reverse_preference"] += reverse
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the V3 expanded gate's presentation-count erratum"
    )
    parser.add_argument("--original-gate", type=Path, required=True)
    parser.add_argument("--v3-run", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = json.loads(args.original_gate.read_text(encoding="utf-8"))
    run = json.loads(args.v3_run.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    gate_source = Path("scripts/check_route_a_v3_expanded_gate.py")
    train_source = Path("scripts/train_route_a_preference.py")
    source_hashes = protocol.get("source_sha256", {})
    failed_checks = sorted(
        key for key, value in original.get("checks", {}).items() if not value
    )
    expected = expected_balanced_counts(
        int(run["presentations_per_epoch"]), int(run["epochs"])
    )
    actual = {
        str(key): int(value)
        for key, value in run.get("presented_decisions", {}).items()
    }
    checks = {
        "original_gate_preserved_as_failure": original.get("passed") is False,
        "only_failed_check_is_bookkeeping": failed_checks
        == ["v3_balanced_causal_presentations"],
        "all_substantive_original_checks_pass": all(
            value
            for key, value in original.get("checks", {}).items()
            if key != "v3_balanced_causal_presentations"
        ),
        "protocol_was_frozen_before_v3": protocol.get("status")
        == "frozen_before_v3_outcomes",
        "original_gate_source_matches_protocol": source_hashes.get(str(gate_source))
        == file_sha256(gate_source),
        "training_source_matches_protocol": source_hashes.get(str(train_source))
        == file_sha256(train_source),
        "matched_total_budget_is_unchanged": run.get("presentations_per_epoch")
        == 255
        and run.get("active_event_presentations") == 765
        and sum(actual.values()) == 765,
        "deterministic_formula_gives_383_382": expected
        == {"rescue_preference": 383, "reverse_preference": 382},
        "actual_counts_match_deterministic_formula": actual == expected,
        "no_metric_or_threshold_changed": original.get("accuracy_improvement")
        == 0.1875
        and protocol.get("gate_thresholds", {}).get(
            "min_accuracy_improvement"
        )
        == 0.10,
    }
    passed = all(checks.values())
    result = {
        "passed": passed,
        "status": "audited_arithmetic_erratum",
        "stage": "route_a_seed42_v3_expanded_preference_gate_erratum",
        "checks": checks,
        "original_failed_checks": failed_checks,
        "original_gate_sha256": file_sha256(args.original_gate),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "erratum": {
            "incorrect_expected_counts": {
                "rescue_preference": 384,
                "reverse_preference": 381,
            },
            "correct_expected_counts": expected,
            "reason": (
                "three odd 255-presentation epochs alternate the extra sample "
                "as rescue, reverse, rescue"
            ),
            "training_rerun_required": False,
            "metrics_changed": False,
            "thresholds_changed": False,
        },
        "mask_causal_accuracy": original["mask_causal_accuracy"],
        "v3_causal_accuracy": original["v3_causal_accuracy"],
        "accuracy_improvement": original["accuracy_improvement"],
        "mask_reverse_accuracy": original["mask_reverse_accuracy"],
        "v3_reverse_accuracy": original["v3_reverse_accuracy"],
        "scope": original["scope"],
        "next_step": (
            "run paired controlled-state AppWorld dev evaluation"
            if passed
            else "stop and inspect erratum integrity"
        ),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
