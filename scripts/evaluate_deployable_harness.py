#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from environments.api_bank import (
    APIBankControlledEnv,
    DeployableAPIBankHarness,
    public_harness_observation,
)
from environments.api_bank.correction_generator import FrozenModelCorrectionGenerator
from environments.api_bank.adapter import canonical_action
from rescuecredit.logging import write_json


def read_tasks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mutation_cases(expected: dict[str, Any], required: list[str]) -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = []
    arguments = dict(expected.get("arguments", {}))
    for parameter in required:
        if parameter in arguments:
            missing = copy.deepcopy(expected)
            missing["arguments"].pop(parameter, None)
            cases.append((f"missing:{parameter}", missing))
            wrong = copy.deepcopy(expected)
            wrong["arguments"][parameter] = "__unsupported_visible_value__"
            cases.append((f"wrong:{parameter}", wrong))
    cases.append(("unknown_tool", {"tool": "__UNKNOWN_TOOL__", "arguments": {}}))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline quality audit for the reference-free deployable harness")
    parser.add_argument("--tasks", type=Path, default=Path("data/api_bank_controlled_v1/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deployable_harness_audit"))
    parser.add_argument("--max-tasks", type=int, default=0, help="0 evaluates the full split")
    parser.add_argument("--min-coverage", type=float, default=0.10)
    parser.add_argument("--min-correction-precision", type=float, default=0.90)
    parser.add_argument("--min-single-step-rescue-rate", type=float, default=0.10)
    parser.add_argument("--max-harm-rate", type=float, default=0.01)
    parser.add_argument("--model", default=None, help="optional frozen local model for missing-argument repair")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generator-max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    tasks = read_tasks(args.tasks)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "case_results.jsonl"
    correction_generator = None
    if args.model:
        correction_generator = FrozenModelCorrectionGenerator(
            args.model,
            revision=args.model_revision,
            device=args.device,
            max_new_tokens=args.generator_max_new_tokens,
        )
    harness = DeployableAPIBankHarness("H3", correction_generator=correction_generator)

    corrupt_total = 0
    corrections = 0
    correct_corrections = 0
    single_step_rescues = 0
    clean_total = 0
    harmful_clean_changes = 0
    unknown_semantics = 0
    records: list[dict[str, Any]] = []

    for task in tasks:
        env = APIBankControlledEnv(task.get("max_steps", 12))
        env.reset(task, seed=0)
        previous_tool_result: dict[str, Any] | None = None
        schemas = {str(tool.get("name", "")): tool for tool in task.get("available_tools", [])}
        for action_index, expected in enumerate(task.get("reference_actions", [])):
            observation = public_harness_observation(env.observation())
            expected_canonical = canonical_action(expected)

            clean_total += 1
            clean_executed, clean_decision = harness.execute(observation, copy.deepcopy(expected), previous_tool_result)
            if clean_decision.changes_execution and canonical_action(clean_executed) != expected_canonical:
                harmful_clean_changes += 1
            records.append(
                {
                    "task_id": task.get("task_id"),
                    "action_index": action_index,
                    "case": "clean",
                    "changed": clean_decision.changes_execution,
                    "correct_after": canonical_action(clean_executed) == expected_canonical,
                    "reference_scope": "offline_evaluation_only",
                }
            )

            schema = schemas.get(str(expected.get("tool", "")), {})
            for case_name, proposal in mutation_cases(expected, [str(x) for x in schema.get("required", [])]):
                corrupt_total += 1
                a_validity = harness.validator.validate(observation, proposal, previous_tool_result)
                executed, decision = harness.execute(observation, proposal, previous_tool_result)
                corrected = decision.corrected_action
                b_validity = (
                    harness.validator.validate(observation, corrected, previous_tool_result) if corrected else None
                )
                if a_validity.semantic_valid == "unknown" or (
                    b_validity is not None and b_validity.semantic_valid == "unknown"
                ):
                    unknown_semantics += 1
                if decision.changes_execution and corrected is not None:
                    corrections += 1
                    is_correct = canonical_action(executed) == expected_canonical
                    correct_corrections += int(is_correct)
                    single_step_rescues += int(is_correct and canonical_action(proposal) != expected_canonical)
                records.append(
                    {
                        "task_id": task.get("task_id"),
                        "action_index": action_index,
                        "case": case_name,
                        "a_executable_valid": a_validity.executable_valid,
                        "a_semantic_valid": a_validity.semantic_valid,
                        "b_executable_valid": b_validity.executable_valid if b_validity else None,
                        "b_semantic_valid": b_validity.semantic_valid if b_validity else None,
                        "changed": decision.changes_execution,
                        "patch_id": decision.patch_id,
                        "correct_after": canonical_action(executed) == expected_canonical,
                        "reference_scope": "offline_evaluation_only",
                    }
                )

            _, _, _, info = env.step(copy.deepcopy(expected))
            previous_tool_result = info.get("tool_result")

    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    coverage = corrections / max(1, corrupt_total)
    precision = correct_corrections / max(1, corrections)
    rescue_rate = single_step_rescues / max(1, corrupt_total)
    harm_rate = harmful_clean_changes / max(1, clean_total)
    metrics = {
        "tasks": len(tasks),
        "clean_cases": clean_total,
        "corrupt_cases": corrupt_total,
        "corrections": corrections,
        "correct_corrections": correct_corrections,
        "coverage": coverage,
        "correction_precision": precision,
        "single_step_rescue_rate": rescue_rate,
        "harm_rate": harm_rate,
        "unknown_semantic_cases": unknown_semantics,
        "reference_boundary": {
            "harness_inputs": ["user_goal", "visible_state", "tool_schema", "previous_tool_result", "proposal"],
            "reference_actions": "offline case construction and scoring only",
        },
        "correction_generator": args.model or "visible_rules_only",
    }
    gate = {
        "passed": bool(
            coverage >= args.min_coverage
            and precision >= args.min_correction_precision
            and rescue_rate >= args.min_single_step_rescue_rate
            and harm_rate <= args.max_harm_rate
        ),
        "thresholds": {
            "min_coverage": args.min_coverage,
            "min_correction_precision": args.min_correction_precision,
            "min_single_step_rescue_rate": args.min_single_step_rescue_rate,
            "max_harm_rate": args.max_harm_rate,
        },
        "metrics": metrics,
        "next_step": "implement RescueCredit-v2 loss" if coverage >= args.min_coverage and precision >= args.min_correction_precision and rescue_rate >= args.min_single_step_rescue_rate and harm_rate <= args.max_harm_rate else "improve deployable harness before training",
    }
    write_json(args.output_dir / "harness_metrics.json", metrics)
    write_json(args.output_dir / "quality_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
