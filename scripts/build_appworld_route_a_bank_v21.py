#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from environments.appworld.deployable import AppWorldCandidateHarness
from rescuecredit.frozen_bank import (
    SCHEMA_VERSION,
    digest,
    event_id,
    file_sha256,
    validate_public_record,
    write_jsonl,
)
from rescuecredit.logging import write_json
try:
    from audit_appworld_deployable_harness import (
        OpenAPISchemaIndex,
        SelectorWorker,
        _canonical,
        _is_supervisor,
        _merge_receipt,
        _receipt,
        _render_rest_call,
    )
except ModuleNotFoundError:
    from scripts.audit_appworld_deployable_harness import (
        OpenAPISchemaIndex,
        SelectorWorker,
        _canonical,
        _is_supervisor,
        _merge_receipt,
        _receipt,
        _render_rest_call,
    )


def _compatible_alternative(selected: Any, candidate: Any) -> bool:
    if isinstance(selected, bool) or isinstance(candidate, bool):
        return type(selected) is type(candidate)
    if isinstance(selected, (int, float)) and isinstance(candidate, (int, float)):
        return True
    return type(selected) is type(candidate)


def _prompt(
    instruction: str,
    receipts: dict[str, Any] | None,
    schema: dict[str, Any],
    proposal: dict[str, Any],
) -> str:
    visible = {
        "task_instruction": instruction,
        "previous_visible_receipts": receipts,
        "public_openapi_schema": schema,
        "proposal": proposal,
    }
    return (
        "Repair missing or unsupported required arguments using only visible evidence. "
        "Return one complete JSON tool call.\n"
        + json.dumps(visible, ensure_ascii=False, sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the train-only multi-site Route-A V2.1 correction bank"
    )
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--subset", choices=["train"], default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-missing-per-variant", type=int, default=3)
    parser.add_argument("--max-variants-per-call", type=int, default=20)
    parser.add_argument("--max-cases-per-task", type=int, default=80)
    parser.add_argument("--max-wrong-value-variants-per-field", type=int, default=2)
    parser.add_argument("--selector-python", type=Path, required=True)
    parser.add_argument("--selector-script", type=Path, required=True)
    parser.add_argument("--selector-model", type=Path, required=True)
    parser.add_argument("--selector-device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.max_missing_per_variant < 1:
        raise ValueError("--max-missing-per-variant must be positive")

    root = args.appworld_root.resolve()
    os.environ["APPWORLD_ROOT"] = str(root)
    from appworld import AppWorld, load_task_ids, update_root

    update_root(str(root))
    task_ids = list(load_task_ids("train"))[args.offset : args.offset + args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema_index = OpenAPISchemaIndex(root / "data" / "api_docs" / "openapi")
    selector = SelectorWorker(
        args.selector_python,
        args.selector_script,
        args.selector_model,
        args.selector_device,
        args.output_dir / "selector_stderr.log",
    )
    harness = AppWorldCandidateHarness(selector, min_selector_candidates=1)
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    reference_failures = 0
    reference_calls = 0
    task_event_counts: Counter[str] = Counter()
    variant_size_counts: Counter[int] = Counter()
    variant_kind_counts: Counter[str] = Counter()
    started = time.time()
    try:
        for task_offset, task_id in enumerate(task_ids):
            world = AppWorld(
                task_id=task_id,
                experiment_name=f"rescuecredit_route_a_v21_bank_{args.seed}_{task_offset}",
                ground_truth_mode="full",
                raise_on_failure=False,
                random_seed=args.seed + task_offset,
            )
            try:
                instruction = str(world.task.instruction)
                calls = list(getattr(world.task.ground_truth, "api_calls", []) or [])
                receipts: dict[str, Any] | None = None
                task_cases = 0
                for call_index, call in enumerate(calls):
                    expected = _canonical(call)
                    schema = schema_index.schema_for(call)
                    arguments = dict(expected.get("arguments", {}))
                    required = sorted(
                        parameter
                        for parameter in list((schema or {}).get("required_fields", []))
                        if parameter in arguments
                    )
                    repair_by_parameter: dict[str, dict[str, Any]] = {}
                    if not _is_supervisor(call) and schema is not None and required:
                        # Select each missing value once. Multi-field variants compose only
                        # these independently reference-free visible-evidence repairs.
                        for parameter in required:
                            single = copy.deepcopy(expected)
                            single["arguments"].pop(parameter, None)
                            repaired, decision = harness.repair(
                                instruction,
                                receipts,
                                single,
                                required,
                                public_schema=schema,
                            )
                            if not decision.changed or parameter not in repaired.get(
                                "arguments", {}
                            ):
                                continue
                            repair_by_parameter[parameter] = {
                                "value": copy.deepcopy(repaired["arguments"][parameter]),
                                "selected_by": decision.selected_by,
                                "candidate_count": decision.candidate_count,
                                "sources": list(decision.selected_sources),
                                "origins": list(decision.selected_origins),
                                "candidate_details": harness.candidate_details(
                                    instruction, receipts, parameter
                                ),
                            }

                        eligible = sorted(repair_by_parameter)
                        call_variants = 0
                        for size in range(
                            1, min(args.max_missing_per_variant, len(eligible)) + 1
                        ):
                            for missing in itertools.combinations(eligible, size):
                                if (
                                    call_variants >= args.max_variants_per_call
                                    or task_cases >= args.max_cases_per_task
                                ):
                                    break
                                proposal = copy.deepcopy(expected)
                                for parameter in missing:
                                    proposal["arguments"].pop(parameter, None)
                                repaired = copy.deepcopy(proposal)
                                for parameter in missing:
                                    repaired["arguments"][parameter] = copy.deepcopy(
                                        repair_by_parameter[parameter]["value"]
                                    )
                                if repaired == proposal:
                                    continue
                                parameter_key = "+".join(missing)
                                eid = event_id(
                                    task_id=str(task_id),
                                    call_index=call_index,
                                    parameter=parameter_key,
                                    proposal=proposal,
                                )
                                public = {
                                    "schema_version": SCHEMA_VERSION,
                                    "event_id": eid,
                                    "split": "train",
                                    "task_id": str(task_id),
                                    "call_index": call_index,
                                    "prompt": _prompt(
                                        instruction, receipts, schema, proposal
                                    ),
                                    "action_a": proposal,
                                    "action_b": repaired,
                                    "patch_id": (
                                        "visible_candidate_repair"
                                        if size == 1
                                        else "visible_multi_candidate_repair"
                                    ),
                                    "missing_parameters": list(missing),
                                    "variant_size": size,
                                    "variant_kind": "missing_required_arguments",
                                    "provenance": {
                                        "repair_composition": (
                                            "independent_single_field_visible_repairs"
                                        ),
                                        "reference_free_candidate_selection": True,
                                        "fields": {
                                            parameter: repair_by_parameter[parameter]
                                            for parameter in missing
                                        },
                                    },
                                    "training_contract": {
                                        "shared_identically_by": [
                                            "mask_correction",
                                            "rescuecredit_v2_1",
                                        ],
                                        "offline_audit_labels_accessible": False,
                                    },
                                }
                                validate_public_record(public)
                                public_rows.append(public)
                                private_rows.append(
                                    {
                                        "event_id": eid,
                                        "task_id": str(task_id),
                                        "variant_size": size,
                                        "proposal_matches_reference": proposal == expected,
                                        "correction_matches_reference": repaired == expected,
                                        "offline_class": (
                                            "reference_matching_correction"
                                            if repaired == expected
                                            else "nonmatching_correction"
                                        ),
                                        "reference_scope": "offline_scoring_only",
                                    }
                                )
                                task_cases += 1
                                call_variants += 1
                                task_event_counts[str(task_id)] += 1
                                variant_size_counts[size] += 1
                                variant_kind_counts["missing_required_arguments"] += 1
                            if (
                                call_variants >= args.max_variants_per_call
                                or task_cases >= args.max_cases_per_task
                            ):
                                break

                        # Add schema-valid wrong-value proposals from other visible
                        # candidates. B remains the same independently selected,
                        # strongly supported reference-free Harness repair.
                        for parameter in eligible:
                            if (
                                call_variants >= args.max_variants_per_call
                                or task_cases >= args.max_cases_per_task
                            ):
                                break
                            selected = repair_by_parameter[parameter]["value"]
                            alternatives = [
                                detail
                                for detail in repair_by_parameter[parameter][
                                    "candidate_details"
                                ]
                                if detail["value"] != selected
                                and _compatible_alternative(selected, detail["value"])
                            ][: args.max_wrong_value_variants_per_field]
                            for alternative_index, alternative in enumerate(alternatives):
                                if (
                                    call_variants >= args.max_variants_per_call
                                    or task_cases >= args.max_cases_per_task
                                ):
                                    break
                                proposal = copy.deepcopy(expected)
                                proposal["arguments"][parameter] = copy.deepcopy(
                                    alternative["value"]
                                )
                                repaired = copy.deepcopy(expected)
                                repaired["arguments"][parameter] = copy.deepcopy(selected)
                                if repaired == proposal:
                                    continue
                                eid = event_id(
                                    task_id=str(task_id),
                                    call_index=call_index,
                                    parameter=(
                                        f"wrong_value:{parameter}:{alternative_index}"
                                    ),
                                    proposal=proposal,
                                )
                                public = {
                                    "schema_version": SCHEMA_VERSION,
                                    "event_id": eid,
                                    "split": "train",
                                    "task_id": str(task_id),
                                    "call_index": call_index,
                                    "prompt": _prompt(
                                        instruction, receipts, schema, proposal
                                    ),
                                    "action_a": proposal,
                                    "action_b": repaired,
                                    "patch_id": "visible_argument_value_repair",
                                    "missing_parameters": [],
                                    "repaired_parameters": [parameter],
                                    "variant_size": 1,
                                    "variant_kind": "wrong_visible_candidate_value",
                                    "provenance": {
                                        "repair_composition": (
                                            "single_field_visible_evidence_replacement"
                                        ),
                                        "reference_free_candidate_selection": True,
                                        "selected_repair": repair_by_parameter[parameter],
                                        "proposal_alternative": {
                                            "sources": list(alternative["sources"]),
                                            "origins": list(alternative["origins"]),
                                        },
                                    },
                                    "training_contract": {
                                        "shared_identically_by": [
                                            "mask_correction",
                                            "rescuecredit_v2_1",
                                        ],
                                        "offline_audit_labels_accessible": False,
                                    },
                                }
                                # Candidate values and origins are visible inputs; the
                                # protected expected action is used only below for audit.
                                validate_public_record(public)
                                public_rows.append(public)
                                private_rows.append(
                                    {
                                        "event_id": eid,
                                        "task_id": str(task_id),
                                        "variant_size": 1,
                                        "variant_kind": "wrong_visible_candidate_value",
                                        "proposal_matches_reference": proposal == expected,
                                        "correction_matches_reference": repaired == expected,
                                        "offline_class": (
                                            "reference_matching_correction"
                                            if repaired == expected
                                            else "nonmatching_correction"
                                        ),
                                        "reference_scope": "offline_scoring_only",
                                    }
                                )
                                task_cases += 1
                                call_variants += 1
                                task_event_counts[str(task_id)] += 1
                                variant_size_counts[1] += 1
                                variant_kind_counts[
                                    "wrong_visible_candidate_value"
                                ] += 1
                    output = str(world.execute(_render_rest_call(call)))
                    visible_receipt = _receipt(output)
                    reference_calls += 1
                    reference_failures += int(visible_receipt.get("status") == "error")
                    receipts = _merge_receipt(receipts, visible_receipt)
            finally:
                world.close()
            print(
                json.dumps(
                    {
                        "progress": f"{task_offset + 1}/{len(task_ids)}",
                        "events": len(public_rows),
                        "tasks_with_events": len(task_event_counts),
                    }
                ),
                flush=True,
            )
    finally:
        selector.close()

    public_rows.sort(key=lambda row: row["event_id"])
    private_rows.sort(key=lambda row: row["event_id"])
    public_path = args.output_dir / "correction_bank.public.jsonl"
    private_path = args.output_dir / "offline_audit.private.jsonl"
    write_jsonl(public_path, public_rows)
    write_jsonl(private_path, private_rows)
    matching = sum(row["correction_matches_reference"] for row in private_rows)
    manifest = {
        "schema_version": "rescuecredit.route_a_bank.v2.1b",
        "status": "frozen",
        "split": "train",
        "task_offset": args.offset,
        "tasks": len(task_ids),
        "tasks_with_events": len(task_event_counts),
        "events": len(public_rows),
        "variant_size_counts": dict(variant_size_counts),
        "variant_kind_counts": dict(variant_kind_counts),
        "reference_matching_corrections": matching,
        "nonmatching_corrections": len(private_rows) - matching,
        "public_bank_sha256": file_sha256(public_path),
        "private_audit_sha256": file_sha256(private_path),
        "event_set_hash": digest([row["event_id"] for row in public_rows]),
        "reference_execution_failures": reference_failures,
        "reference_execution_failure_rate": reference_failures
        / max(1, reference_calls),
        "generation": {
            "max_missing_per_variant": args.max_missing_per_variant,
            "max_variants_per_call": args.max_variants_per_call,
            "max_cases_per_task": args.max_cases_per_task,
            "max_wrong_value_variants_per_field": (
                args.max_wrong_value_variants_per_field
            ),
        },
        "hard_boundary": {
            "training_reads": "correction_bank.public.jsonl only",
            "training_forbidden": "offline_audit.private.jsonl",
            "dev_or_test_events": 0,
        },
        "limitations": [
            "A is a controlled one-to-three-required-argument corruption",
            "A may instead contain a type-compatible alternative visible value",
            "B composes independently selected visible-evidence repairs",
            "offline exact-match labels are diagnostics and never training targets",
            "causal deltas are attached only by paired official AppWorld evaluation",
        ],
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
