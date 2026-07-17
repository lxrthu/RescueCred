#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from environments.api_bank import DeployableAPIBankHarness, VisibleContextSemanticValidator
from environments.api_bank.adapter import canonical_action
from rescuecredit.logging import write_json


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def digest_records(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def unique_visible_identifier(task: dict[str, Any]) -> bool:
    references = task.get("reference_actions", [])
    if len(references) != 1:
        return False
    id_values = [
        str(value)
        for key, value in dict(references[0].get("arguments", {})).items()
        if str(key).endswith("_id")
    ]
    visible = list(dict.fromkeys(re.findall(r"(?<!\d)\d{6,}(?!\d)", task.get("user_goal", ""))))
    return len(id_values) == 1 and len(visible) == 1 and id_values[0] == visible[0]


def unambiguous_two_step_dependency(task: dict[str, Any]) -> bool:
    references = task.get("reference_actions", [])
    tools = {tool["name"]: tool for tool in task.get("available_tools", [])}
    receipts = task.get("reference_tool_receipts", [])
    if len(references) != 2 or len(tools) != 2 or len(receipts) < 1:
        return False
    producer, consumer = references
    if producer.get("tool") not in tools or consumer.get("tool") not in tools:
        return False
    output_names = set(tools[producer["tool"]].get("output_parameters", {}))
    required_names = set(tools[consumer["tool"]].get("required", []))
    dependency = output_names & required_names
    if len(dependency) != 1:
        return False
    dependency_name = next(iter(dependency))
    if dependency_name not in receipts[0]:
        return False
    if dict(consumer.get("arguments", {})).get(dependency_name) != receipts[0][dependency_name]:
        return False
    observation = {
        "user_goal": task.get("user_goal", ""),
        "available_tools": task.get("available_tools", []),
        "call_index": 0,
        "success_predicate_satisfied": False,
    }
    validity = VisibleContextSemanticValidator().validate(observation, producer)
    return validity.semantic_valid == "true"


def select_task(task: dict[str, Any]) -> str | None:
    if unique_visible_identifier(task):
        return "single_step_unique_identifier"
    if unambiguous_two_step_dependency(task):
        return "two_step_visible_dependency"
    return None


def repair_contract_holds(task: dict[str, Any], subset_type: str) -> bool:
    """Offline integrity check for the runtime reference-free repair rule.

    References are used here only to construct/scored a deterministic mutation.
    They are never added to the Harness observation or available at runtime.
    """

    references = task.get("reference_actions", [])
    observation = {
        "user_goal": task.get("user_goal", ""),
        "available_tools": task.get("available_tools", []),
        "call_index": 0,
        "success_predicate_satisfied": False,
    }
    if subset_type == "single_step_unique_identifier":
        expected = canonical_action(references[0])
        proposal = canonical_action(references[0])
        arguments = dict(proposal.get("arguments", {}))
        id_fields = [name for name in arguments if str(name).endswith("_id")]
        if len(id_fields) != 1:
            return False
        arguments[id_fields[0]] = "999999999999"
        proposal["arguments"] = arguments
    elif subset_type == "two_step_visible_dependency":
        producer, consumer = references
        tools = {tool["name"]: tool for tool in task.get("available_tools", [])}
        dependency = set(tools[producer["tool"]].get("output_parameters", {})) & set(
            tools[consumer["tool"]].get("required", [])
        )
        if len(dependency) != 1:
            return False
        expected = canonical_action(producer)
        proposal = canonical_action(consumer)
        arguments = dict(proposal.get("arguments", {}))
        arguments[next(iter(dependency))] = "runtime_value_not_yet_observed"
        proposal["arguments"] = arguments
    else:
        return False
    executed, decision = DeployableAPIBankHarness("H3").execute(observation, proposal)
    return decision.changes_execution and canonical_action(executed) == expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/api_bank_controlled_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/api_bank_causal_v1"))
    args = parser.parse_args()

    source_manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    source_catalog = json.loads((args.input_dir / "api_catalog.json").read_text(encoding="utf-8"))
    source_tasks = [
        json.loads(line)
        for line in (args.input_dir / "tasks_all.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    structural_counts: dict[str, int] = {}
    contract_rejected: dict[str, int] = {}
    for source_task in source_tasks:
        task = copy.deepcopy(source_task)
        task["available_tools"] = [
            copy.deepcopy(source_catalog.get(tool.get("name"), tool))
            for tool in task.get("available_tools", [])
        ]
        subset_type = select_task(task)
        if subset_type is None:
            continue
        structural_counts[subset_type] = structural_counts.get(subset_type, 0) + 1
        if not repair_contract_holds(task, subset_type):
            contract_rejected[subset_type] = contract_rejected.get(subset_type, 0) + 1
            continue
        record = dict(task)
        record["source_task_id"] = task["task_id"]
        record["causal_subset_type"] = subset_type
        selected.append(record)
        type_counts[subset_type] = type_counts.get(subset_type, 0) + 1

    selected.sort(
        key=lambda task: hashlib.sha256(
            f"api-bank-causal-v1:{task['source_sample_id']}".encode()
        ).hexdigest()
    )
    count = len(selected)
    train_end = round(0.70 * count)
    dev_end = train_end + round(0.15 * count)
    splits = {
        "train": selected[:train_end],
        "dev": selected[train_end:dev_end],
        "test_id": selected[dev_end:],
    }
    for split, records in splits.items():
        for index, task in enumerate(records):
            task["task_id"] = f"apibank_causal_{split}_{index:06d}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = [task for split in ("train", "dev", "test_id") for task in splits[split]]
    write_jsonl(args.output_dir / "tasks_all.jsonl", all_records)
    for split, records in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", records)
    write_jsonl(args.output_dir / "full_shadow_eval.jsonl", splits["dev"] + splits["test_id"])
    (args.output_dir / "api_catalog.json").write_text(
        (args.input_dir / "api_catalog.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    manifest = {
        "name": "API-Bank causal-identifiable subset",
        "version": "api_bank_causal_v1",
        "not_official_leaderboard": True,
        "source_dataset": str(args.input_dir),
        "source_split_hashes": source_manifest["split_hashes"],
        "selection_rules": [
            "single reference action with one uniquely visible *_id value",
            "exactly two tools/actions where action 1 uniquely produces one required input of action 2",
            "producer action inputs must be semantically supported by visible user context",
            "deterministic mutation must be corrected to the reference action by the runtime reference-free Harness",
        ],
        "selection_reference_usage": "offline integrity scoring only; forbidden in Harness observations and training labels",
        "reference_actions_runtime_visibility": "forbidden",
        "counts": {split: len(records) for split, records in splits.items()},
        "selected_total": count,
        "type_counts": type_counts,
        "structural_candidate_counts": structural_counts,
        "contract_rejected_counts": contract_rejected,
        "split_hashes": {split: digest_records(records) for split, records in splits.items()},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
