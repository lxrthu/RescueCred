#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

from environments.appworld.deployable import AppWorldCandidateHarness
from rescuecredit.frozen_bank import digest, file_sha256, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_task_eval import event_set_hash
from audit_appworld_deployable_harness import (
    OpenAPISchemaIndex,
    SelectorWorker,
    _canonical,
    _is_supervisor,
    _merge_receipt,
    _receipt,
    _render_rest_call,
)
from build_appworld_route_a_bank import _prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--subset", choices=["dev"], default="dev")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=57)
    parser.add_argument("--seed", type=int, default=42)
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
    task_ids = list(load_task_ids("dev"))[args.offset : args.offset + args.limit]
    schema_index = OpenAPISchemaIndex(root / "data" / "api_docs" / "openapi")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selector = SelectorWorker(
        args.selector_python,
        args.selector_script,
        args.selector_model,
        args.selector_device,
        args.output_dir / "selector_stderr.log",
    )
    harness = AppWorldCandidateHarness(selector, min_selector_candidates=1)
    rows: list[dict] = []
    reference_failures = 0
    repair_abstentions = 0
    started = time.time()
    try:
        for task_index, task_id in enumerate(task_ids):
            world = AppWorld(
                task_id=task_id,
                experiment_name=f"route_a_dev_event_builder_v2_{args.seed}_{task_index}",
                ground_truth_mode="full",
                raise_on_failure=False,
                random_seed=args.seed + task_index,
            )
            receipts = None
            try:
                instruction = str(world.task.instruction)
                calls = list(getattr(world.task.ground_truth, "api_calls", []) or [])
                for call_index, call in enumerate(calls):
                    expected = _canonical(call)
                    schema = schema_index.schema_for(call)
                    arguments = dict(expected.get("arguments", {}))
                    required = list((schema or {}).get("required_fields", []))
                    eligible = (
                        not _is_supervisor(call)
                        and schema is not None
                        and bool(arguments)
                        and bool(required)
                    )
                    if eligible:
                        parameter = next(
                            (name for name in required if name in arguments), None
                        )
                        if parameter is not None:
                            proposal = copy.deepcopy(expected)
                            proposal["arguments"].pop(parameter, None)
                            repaired, decision = harness.repair(
                                instruction,
                                receipts,
                                proposal,
                                required,
                                public_schema=schema,
                            )
                            if decision.changed and repaired != proposal:
                                eid = digest(
                                    {
                                        "split": "dev",
                                        "task_id": str(task_id),
                                        "call_index": call_index,
                                        "missing_parameter": parameter,
                                        "proposal": proposal,
                                        "candidate": repaired,
                                    }
                                )
                                rows.append(
                                    {
                                        "event_id": eid,
                                        "split": "dev",
                                        "task_id": str(task_id),
                                        "task_index": task_index,
                                        "call_index": call_index,
                                        "missing_parameter": parameter,
                                        "prompt": _prompt(
                                            instruction, receipts, schema, proposal
                                        ),
                                        "action_a": proposal,
                                        "action_b": repaired,
                                        "candidate_provenance": {
                                            "selected_by": decision.selected_by,
                                            "candidate_count": decision.candidate_count,
                                            "selected_sources": list(
                                                decision.selected_sources
                                            ),
                                            "selected_origins": list(
                                                decision.selected_origins
                                            ),
                                        },
                                        "expected_action_hash": digest(expected),
                                        "adapter_input_contract": [
                                            "public_prompt",
                                            "action_a",
                                            "action_b",
                                        ],
                                        "protected_reference_actions_or_labels_in_model_input": False,
                                    }
                                )
                                break
                            repair_abstentions += 1
                    output = str(world.execute(_render_rest_call(call)))
                    receipt = _receipt(output)
                    reference_failures += int(receipt.get("status") == "error")
                    receipts = _merge_receipt(receipts, receipt)
            finally:
                world.close()
    finally:
        selector.close()

    event_path = args.output_dir / "dev_events.public.jsonl"
    write_jsonl(event_path, rows)
    manifest = {
        "status": "frozen",
        "split": "dev",
        "seed": args.seed,
        "tasks_requested": len(task_ids),
        "events": len(rows),
        "event_set_hash": event_set_hash(rows),
        "event_file_sha256": file_sha256(event_path),
        "reference_replay_failures_before_selected_event": reference_failures,
        "reference_free_candidate_repairs": len(rows),
        "candidate_repair_abstentions": repair_abstentions,
        "training_access": False,
        "test_split_access": False,
        "protected_reference_actions_or_labels_exported": False,
        "candidate_generation": "reference-free visible-candidate harness",
        "reference_role": "offline event construction and scoring only",
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
