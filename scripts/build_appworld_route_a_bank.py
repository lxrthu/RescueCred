#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import time
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
from audit_appworld_deployable_harness import (
    OpenAPISchemaIndex,
    SelectorWorker,
    _canonical,
    _is_supervisor,
    _merge_receipt,
    _receipt,
    _render_rest_call,
)


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
        "Repair the proposed tool call using only the visible context. "
        "Return one JSON tool call.\n"
        + json.dumps(visible, ensure_ascii=False, sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen, train-only Route-A AppWorld correction bank"
    )
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--subset", choices=["train"], default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases-per-task", type=int, default=20)
    parser.add_argument("--selector-python", type=Path, required=True)
    parser.add_argument("--selector-script", type=Path, required=True)
    parser.add_argument("--selector-model", type=Path, required=True)
    parser.add_argument("--selector-device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

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
    started = time.time()
    try:
        for task_offset, task_id in enumerate(task_ids):
            world = AppWorld(
                task_id=task_id,
                experiment_name=f"rescuecredit_route_a_bank_{args.seed}_{task_offset}",
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
                    required = list((schema or {}).get("required_fields", []))
                    arguments = dict(expected.get("arguments", {}))
                    if (
                        not _is_supervisor(call)
                        and schema is not None
                        and arguments
                        and required
                        and task_cases < args.max_cases_per_task
                    ):
                        for parameter in required:
                            if task_cases >= args.max_cases_per_task:
                                break
                            if parameter not in arguments:
                                continue
                            proposal = copy.deepcopy(expected)
                            proposal["arguments"].pop(parameter, None)
                            repaired, decision = harness.repair(
                                instruction,
                                receipts,
                                proposal,
                                required,
                                public_schema=schema,
                            )
                            task_cases += 1
                            if not decision.changed or repaired == proposal:
                                continue
                            eid = event_id(
                                task_id=str(task_id),
                                call_index=call_index,
                                parameter=parameter,
                                proposal=proposal,
                            )
                            public = {
                                "schema_version": SCHEMA_VERSION,
                                "event_id": eid,
                                "split": "train",
                                "task_id": str(task_id),
                                "call_index": call_index,
                                "prompt": _prompt(instruction, receipts, schema, proposal),
                                "action_a": proposal,
                                "action_b": repaired,
                                "patch_id": decision.patch_id,
                                "provenance": {
                                    "selected_by": decision.selected_by,
                                    "candidate_count": decision.candidate_count,
                                    "selected_sources": list(decision.selected_sources),
                                    "selected_origins": list(decision.selected_origins),
                                    "reference_free_candidate_selection": True,
                                },
                                "training_contract": {
                                    "shared_identically_by": [
                                        "mask_correction",
                                        "rescuecredit_v2",
                                    ],
                                    "offline_audit_labels_accessible": False,
                                },
                            }
                            validate_public_record(public)
                            public_rows.append(public)
                            private_rows.append(
                                {
                                    "event_id": eid,
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
                    output = str(world.execute(_render_rest_call(call)))
                    visible_receipt = _receipt(output)
                    reference_calls += 1
                    reference_failures += int(visible_receipt.get("status") == "error")
                    receipts = _merge_receipt(receipts, visible_receipt)
            finally:
                world.close()
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
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "split": "train",
        "task_offset": args.offset,
        "tasks": len(task_ids),
        "events": len(public_rows),
        "reference_matching_corrections": matching,
        "nonmatching_corrections": len(private_rows) - matching,
        "public_bank_sha256": file_sha256(public_path),
        "private_audit_sha256": file_sha256(private_path),
        "event_set_hash": digest([row["event_id"] for row in public_rows]),
        "reference_execution_failures": reference_failures,
        "reference_execution_failure_rate": reference_failures / max(1, reference_calls),
        "hard_boundary": {
            "training_reads": "correction_bank.public.jsonl only",
            "training_forbidden": "offline_audit.private.jsonl",
            "dev_or_test_events": 0,
        },
        "limitations": [
            "proposal A is a controlled missing-required-argument corruption",
            "offline exact-match labels are evaluation diagnostics, not training targets",
            "causal Shadow deltas are not fabricated by this builder",
        ],
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
