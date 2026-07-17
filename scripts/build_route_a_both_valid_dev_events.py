#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from datetime import date, datetime
from collections import Counter
from pathlib import Path
from typing import Any

from environments.appworld.deployable import AppWorldCandidateHarness
from rescuecredit.frozen_bank import digest, file_sha256, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_task_eval import event_set_hash

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
    from build_appworld_route_a_bank_v21 import _compatible_alternative, _prompt
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
    from scripts.build_appworld_route_a_bank_v21 import (
        _compatible_alternative,
        _prompt,
    )


STRONG_SOURCES = {
    "exact_receipt_key",
    "related_receipt_key",
    "instruction_labeled",
}


def _full_public_schema(
    index: OpenAPISchemaIndex, call: dict[str, Any]
) -> dict[str, Any] | None:
    """Match a route without the legacy GT-present required-field filtering."""
    method = str(call.get("method", "")).lower()
    url = str(call.get("url", "")).split("?", 1)[0]
    parts = [part for part in url.split("/") if part]
    app = parts[0] if parts else ""
    stripped = "/" + "/".join(parts[1:]) if len(parts) > 1 else "/"
    for route_app, route_method, pattern, public_schema in index.routes:
        if route_app == app and route_method == method and (
            pattern.fullmatch(url) or pattern.fullmatch(stripped)
        ):
            return copy.deepcopy(public_schema)
    return None


def _unsupported_schema_keyword(schema: dict[str, Any]) -> bool:
    unsupported = {"$ref", "anyOf", "oneOf", "allOf", "not"}
    if unsupported.intersection(schema):
        return True
    nested: list[Any] = [schema.get("items")]
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        nested.extend(properties.values())
    return any(
        isinstance(item, dict) and _unsupported_schema_keyword(item)
        for item in nested
    )


def _format_valid(value: str, format_name: str) -> bool:
    if format_name == "email":
        return re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value) is not None
    if format_name == "date":
        try:
            date.fromisoformat(value)
            return True
        except ValueError:
            return False
    if format_name == "date-time":
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False
    if format_name == "uuid":
        return re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
            r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}",
            value,
        ) is not None
    return False


def _schema_value_valid(value: Any, schema: dict[str, Any]) -> bool:
    """Validate visible action values against the public OpenAPI subset we use."""
    if not isinstance(schema, dict):
        return True
    if _unsupported_schema_keyword(schema):
        return False
    if value is None:
        return bool(schema.get("nullable")) or schema.get("type") == "null"
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        if not any(_schema_value_valid(value, {**schema, "type": item}) for item in expected_type):
            return False
    elif expected_type == "string":
        if not isinstance(value, str):
            return False
        if len(value) < int(schema.get("minLength", 0)):
            return False
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            return False
        if schema.get("pattern"):
            try:
                if re.search(str(schema["pattern"]), value) is None:
                    return False
            except re.error:
                return False
        if schema.get("format") and not _format_valid(
            value, str(schema["format"])
        ):
            return False
    elif expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return False
    elif expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
    elif expected_type == "boolean" and not isinstance(value, bool):
        return False
    elif expected_type == "array":
        if not isinstance(value, list):
            return False
        if len(value) < int(schema.get("minItems", 0)):
            return False
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return False
        item_schema = schema.get("items", {})
        if not all(_schema_value_valid(item, item_schema) for item in value):
            return False
    elif expected_type == "object":
        if not isinstance(value, dict):
            return False
        required = schema.get("required", [])
        if not all(name in value for name in required):
            return False
        properties = schema.get("properties", {})
        if not all(
            name not in properties or _schema_value_valid(item, properties[name])
            for name, item in value.items()
        ):
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
        exclusive_minimum = schema.get("exclusiveMinimum")
        if exclusive_minimum is True and "minimum" in schema and value <= schema["minimum"]:
            return False
        if not isinstance(exclusive_minimum, bool) and exclusive_minimum is not None and value <= exclusive_minimum:
            return False
        exclusive_maximum = schema.get("exclusiveMaximum")
        if exclusive_maximum is True and "maximum" in schema and value >= schema["maximum"]:
            return False
        if not isinstance(exclusive_maximum, bool) and exclusive_maximum is not None and value >= exclusive_maximum:
            return False
    return True


def _schema_valid(action: dict[str, Any], public_schema: dict[str, Any]) -> bool:
    if public_schema.get("unsupported_schema_keywords"):
        return False
    arguments = action.get("arguments")
    if not isinstance(arguments, dict):
        return False
    required = list(public_schema.get("required_fields", []))
    if not all(name in arguments for name in required):
        return False
    parameter_schemas = public_schema.get("parameter_schemas", {})
    return all(
        _schema_value_valid(value, parameter_schemas.get(name, {}))
        for name, value in arguments.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a frozen AppWorld dev fixture where both candidate actions are schema-complete"
    )
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--subset", choices=["dev"], default="dev")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=57)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=80)
    parser.add_argument("--max-events-per-task", type=int, default=3)
    parser.add_argument("--max-alternatives-per-field", type=int, default=2)
    parser.add_argument("--selector-python", type=Path, required=True)
    parser.add_argument("--selector-script", type=Path, required=True)
    parser.add_argument("--selector-model", type=Path, required=True)
    parser.add_argument("--selector-device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_events, args.max_events_per_task, args.max_alternatives_per_field) < 1:
        raise ValueError("event and alternative caps must be positive")

    root = args.appworld_root.resolve()
    os.environ["APPWORLD_ROOT"] = str(root)
    from appworld import AppWorld, load_task_ids, update_root

    update_root(str(root))
    task_ids = list(load_task_ids("dev"))[args.offset : args.offset + args.limit]
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
    rows: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    reference_calls = 0
    reference_failures = 0
    repair_abstentions = 0
    schema_rejections = 0
    started = time.time()
    try:
        for task_index, task_id in enumerate(task_ids):
            if len(rows) >= args.max_events:
                break
            world = AppWorld(
                task_id=task_id,
                experiment_name=f"route_a_both_valid_dev_builder_{args.seed}_{task_index}",
                ground_truth_mode="full",
                raise_on_failure=False,
                random_seed=args.seed + task_index,
            )
            receipts: dict[str, Any] | None = None
            try:
                instruction = str(world.task.instruction)
                calls = list(getattr(world.task.ground_truth, "api_calls", []) or [])
                for call_index, call in enumerate(calls):
                    if (
                        len(rows) >= args.max_events
                        or task_counts[str(task_id)] >= args.max_events_per_task
                    ):
                        break
                    expected = _canonical(call)
                    schema = _full_public_schema(schema_index, call)
                    arguments = dict(expected.get("arguments", {}))
                    required = sorted(list((schema or {}).get("required_fields", [])))
                    if not _is_supervisor(call) and schema is not None and required:
                        for parameter in required:
                            if (
                                len(rows) >= args.max_events
                                or task_counts[str(task_id)] >= args.max_events_per_task
                            ):
                                break
                            missing = copy.deepcopy(expected)
                            missing["arguments"].pop(parameter, None)
                            repaired, decision = harness.repair(
                                instruction,
                                receipts,
                                missing,
                                required,
                                public_schema=schema,
                            )
                            if not decision.changed or parameter not in repaired.get(
                                "arguments", {}
                            ):
                                repair_abstentions += 1
                                continue
                            selected = copy.deepcopy(repaired["arguments"][parameter])
                            details = harness.candidate_details(
                                instruction, receipts, parameter
                            )
                            alternatives = [
                                detail
                                for detail in details
                                if detail["value"] != selected
                                and _compatible_alternative(selected, detail["value"])
                            ][: args.max_alternatives_per_field]
                            for alternative_index, alternative in enumerate(alternatives):
                                if (
                                    len(rows) >= args.max_events
                                    or task_counts[str(task_id)]
                                    >= args.max_events_per_task
                                ):
                                    break
                                action_a = copy.deepcopy(expected)
                                action_b = copy.deepcopy(expected)
                                action_a["arguments"][parameter] = copy.deepcopy(
                                    alternative["value"]
                                )
                                action_b["arguments"][parameter] = selected
                                if action_a == action_b:
                                    continue
                                a_schema_valid = _schema_valid(action_a, schema)
                                b_schema_valid = _schema_valid(action_b, schema)
                                if not (a_schema_valid and b_schema_valid):
                                    schema_rejections += 1
                                    continue
                                eid = digest(
                                    {
                                        "stage": "route_a_v31_both_valid_dev",
                                        "task_id": str(task_id),
                                        "call_index": call_index,
                                        "parameter": parameter,
                                        "alternative_index": alternative_index,
                                        "action_a": action_a,
                                        "action_b": action_b,
                                    }
                                )
                                selected_sources = list(decision.selected_sources)
                                alternative_sources = list(alternative["sources"])
                                for source in selected_sources + alternative_sources:
                                    source_counts[source] += 1
                                rows.append(
                                    {
                                        "event_id": eid,
                                        "split": "dev",
                                        "task_id": str(task_id),
                                        "task_index": task_index,
                                        "call_index": call_index,
                                        "variant_kind": "visible_candidate_value_pair",
                                        "parameter": parameter,
                                        "prompt": _prompt(
                                            instruction, receipts, schema, action_a
                                        ),
                                        "continuation_context": json.dumps(
                                            {
                                                "task_instruction": instruction,
                                                "previous_visible_receipts": receipts,
                                                "public_openapi_schema": schema,
                                                "repair_parameter": parameter,
                                                "original_visible_proposal": action_a,
                                            },
                                            ensure_ascii=False,
                                            sort_keys=True,
                                        ),
                                        "action_a": action_a,
                                        "action_b": action_b,
                                        "action_a_schema_valid": a_schema_valid,
                                        "action_b_schema_valid": b_schema_valid,
                                        "required_fields": required,
                                        "parameter_schemas": schema.get(
                                            "parameter_schemas", {}
                                        ),
                                        "unsupported_schema_keywords": schema.get(
                                            "unsupported_schema_keywords", []
                                        ),
                                        "candidate_provenance": {
                                            "selected_by": decision.selected_by,
                                            "candidate_count": decision.candidate_count,
                                            "b_sources": selected_sources,
                                            "b_origins": list(
                                                decision.selected_origins
                                            ),
                                            "a_sources": alternative_sources,
                                            "a_origins": list(alternative["origins"]),
                                            "a_has_strong_support": bool(
                                                STRONG_SOURCES.intersection(
                                                    alternative_sources
                                                )
                                            ),
                                            "b_has_strong_support": bool(
                                                STRONG_SOURCES.intersection(
                                                    selected_sources
                                                )
                                            ),
                                        },
                                        "fixture_contract": {
                                            "both_actions_schema_complete": True,
                                            "actions_differ_in_one_visible_candidate_value": True,
                                            "controlled_reference_prefix_and_common_fields": True,
                                            "reference_action_not_used_as_preference_label": True,
                                            "reference_suffix_not_exposed": True,
                                            "business_acceptance_not_used_as_validity_filter": True,
                                            "original_proposal_is_common_pretreatment_context": True,
                                        },
                                        "test_split_access": False,
                                    }
                                )
                                task_counts[str(task_id)] += 1
                    output = str(world.execute(_render_rest_call(call)))
                    receipt = _receipt(output)
                    reference_calls += 1
                    reference_failures += int(receipt.get("status") == "error")
                    receipts = _merge_receipt(receipts, receipt)
            finally:
                world.close()
            print(
                json.dumps(
                    {
                        "progress": f"{task_index + 1}/{len(task_ids)}",
                        "events": len(rows),
                        "tasks_with_events": len(task_counts),
                    }
                ),
                flush=True,
            )
    finally:
        selector.close()

    rows.sort(key=lambda row: row["event_id"])
    event_path = args.output_dir / "both_valid_dev_events.public.jsonl"
    write_jsonl(event_path, rows)
    max_task_events = max(task_counts.values(), default=0)
    manifest = {
        "status": "frozen_before_bounded_outcomes",
        "stage": "route_a_v31_both_valid_dev_events",
        "split": "dev",
        "seed": args.seed,
        "tasks_requested": len(task_ids),
        "tasks_with_events": len(task_counts),
        "events": len(rows),
        "event_set_hash": event_set_hash(rows),
        "event_file_sha256": file_sha256(event_path),
        "variant_kinds": dict(Counter(row["variant_kind"] for row in rows)),
        "max_events_per_task": max_task_events,
        "max_task_event_share": max_task_events / max(1, len(rows)),
        "visible_source_counts": dict(sorted(source_counts.items())),
        "repair_abstentions": repair_abstentions,
        "schema_rejections": schema_rejections,
        "reference_calls": reference_calls,
        "reference_execution_failures": reference_failures,
        "reference_execution_failure_rate": reference_failures
        / max(1, reference_calls),
        "both_actions_schema_complete": all(
            row["action_a_schema_valid"] and row["action_b_schema_valid"]
            for row in rows
        ),
        "training_access": False,
        "test_split_access": False,
        "protected_outcome_labels_exported": False,
        "controlled_reference_fixture_common_fields": True,
        "reference_role": (
            "offline prefix/state and common action fixture construction only; "
            "the original visible proposal A is identical pre-treatment context for "
            "both branches, never a preference label or continuation suffix"
        ),
        "generation": {
            "max_events": args.max_events,
            "max_events_per_task": args.max_events_per_task,
            "max_alternatives_per_field": args.max_alternatives_per_field,
        },
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
