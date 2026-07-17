#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from environments.appworld.deployable import AppWorldCandidateHarness
from rescuecredit.logging import write_json


def _tool_name(call: dict[str, Any]) -> str:
    method = str(call.get("method", "")).lower()
    url = str(call.get("url", ""))
    return f"{method}:{url}"


def _tool_hash(call: dict[str, Any]) -> str:
    return hashlib.sha256(_tool_name(call).encode()).hexdigest()


def _is_supervisor(call: dict[str, Any]) -> bool:
    return str(call.get("url", "")).startswith("/supervisor/")


def _render_rest_call(call: dict[str, Any]) -> str:
    method = str(call.get("method", "")).lower()
    if method not in {"get", "post", "put", "patch", "delete"}:
        raise ValueError("unsupported AppWorld HTTP method")
    url = str(call.get("url", ""))
    if not url.startswith("/"):
        raise ValueError("AppWorld URL must be a local absolute path")
    data = call.get("data", {})
    if not isinstance(data, dict):
        raise TypeError("AppWorld API-call data must be an object")
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return (
        "import json\n"
        f"print(requester.{method}({url!r}, data=json.loads({encoded!r})))"
    )


def _receipt(output: str) -> dict[str, Any]:
    failed = "execution failed" in output.lower() or "traceback" in output.lower()
    if failed:
        return {"status": "error", "feedback": "reference_execution_failed"}
    try:
        payload = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        payload = {"visible_text": output}
    # AppWorld's shell print helper can JSON-encode an already serialized API
    # response. Unwrap bounded nested JSON strings so receipt keys remain
    # available to the deployable validator.
    for _ in range(3):
        if not isinstance(payload, str):
            break
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            break
    if isinstance(payload, dict):
        result = copy.deepcopy(payload)
    else:
        result = {"result": payload}
    result["status"] = "ok"
    return result


def _canonical(call: dict[str, Any]) -> dict[str, Any]:
    data = call.get("data", {})
    if not isinstance(data, dict):
        raise TypeError("AppWorld API-call data must be an object")
    return {
        "tool": _tool_name(call),
        "arguments": copy.deepcopy(data),
    }


def _hash_task(task_id: str) -> str:
    return hashlib.sha256(task_id.encode()).hexdigest()


def _unsupported_openapi_schema_keywords(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"$ref", "anyOf", "oneOf", "allOf", "not"}:
                found.add(key)
            found.update(_unsupported_openapi_schema_keywords(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_unsupported_openapi_schema_keywords(item))
    return found


class OpenAPISchemaIndex:
    def __init__(self, directory: Path) -> None:
        self.routes: list[tuple[str, str, re.Pattern[str], dict[str, Any]]] = []
        for path in sorted(directory.glob("*.json")):
            app = path.stem
            spec = json.loads(path.read_text(encoding="utf-8"))
            for template, path_item in dict(spec.get("paths", {})).items():
                if not isinstance(path_item, dict):
                    continue
                for method in ("get", "post", "put", "patch", "delete"):
                    operation = path_item.get(method)
                    if not isinstance(operation, dict):
                        continue
                    parameters = list(path_item.get("parameters", [])) + list(
                        operation.get("parameters", [])
                    )
                    required = [
                        str(item.get("name"))
                        for item in parameters
                        if isinstance(item, dict)
                        and item.get("required")
                        and item.get("in") != "path"
                        and item.get("name")
                    ]
                    parameter_descriptions: dict[str, str] = {}
                    parameter_schemas: dict[str, Any] = {}
                    unsupported_schema_keywords: set[str] = set()
                    for item in parameters:
                        if not isinstance(item, dict) or not item.get("name"):
                            continue
                        name = str(item["name"])
                        if item.get("description"):
                            parameter_descriptions[name] = str(item["description"])
                        if isinstance(item.get("schema"), dict):
                            parameter_schemas[name] = copy.deepcopy(item["schema"])
                            unsupported_schema_keywords.update(
                                _unsupported_openapi_schema_keywords(item["schema"])
                            )
                    request_body = operation.get("requestBody", {})
                    if isinstance(request_body, dict):
                        content = request_body.get("content", {})
                        if isinstance(content, dict):
                            for media in content.values():
                                schema = media.get("schema", {}) if isinstance(media, dict) else {}
                                if isinstance(schema, dict):
                                    unsupported_schema_keywords.update(
                                        _unsupported_openapi_schema_keywords(schema)
                                    )
                                    required.extend(map(str, schema.get("required", [])))
                                    properties = schema.get("properties", {})
                                    if isinstance(properties, dict):
                                        for name, prop in properties.items():
                                            if not isinstance(prop, dict):
                                                continue
                                            name = str(name)
                                            if prop.get("description"):
                                                parameter_descriptions[name] = str(
                                                    prop["description"]
                                                )
                                            parameter_schemas[name] = copy.deepcopy(prop)
                    pattern = re.compile(
                        "^"
                        + re.sub(r"\\\{[^/{}]+\\\}", r"[^/]+", re.escape(str(template)))
                        + "$"
                    )
                    public_schema = {
                        "required_fields": sorted(set(required)),
                        "tool_summary": str(operation.get("summary", "")),
                        "tool_description": str(operation.get("description", "")),
                        "parameter_descriptions": parameter_descriptions,
                        "parameter_schemas": parameter_schemas,
                        "unsupported_schema_keywords": sorted(
                            unsupported_schema_keywords
                        ),
                    }
                    self.routes.append((app, method, pattern, public_schema))

    def schema_for(self, call: dict[str, Any]) -> dict[str, Any] | None:
        method = str(call.get("method", "")).lower()
        url = str(call.get("url", "")).split("?", 1)[0]
        parts = [part for part in url.split("/") if part]
        app = parts[0] if parts else ""
        stripped = "/" + "/".join(parts[1:]) if len(parts) > 1 else "/"
        data = call.get("data", {})
        if not isinstance(data, dict):
            return None
        for route_app, route_method, pattern, public_schema in self.routes:
            if route_app == app and route_method == method and (
                pattern.fullmatch(url) or pattern.fullmatch(stripped)
            ):
                result = copy.deepcopy(public_schema)
                result["required_fields"] = [
                    name
                    for name in public_schema["required_fields"]
                    if name in data
                ]
                return result
        return None


def _merge_receipt(
    context: dict[str, Any] | None, receipt: dict[str, Any]
) -> dict[str, Any]:
    events = list((context or {}).get("events", []))
    events.append(copy.deepcopy(receipt))
    return {"events": events[-20:]}


class SelectorWorker:
    def __init__(
        self,
        python: Path,
        script: Path,
        model: Path,
        device: str,
        stderr_path: Path,
    ) -> None:
        self.stderr_handle = stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(python),
                str(script),
                "--model",
                str(model),
                "--device",
                device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            bufsize=1,
        )

    def _query(self, payload: dict[str, Any]) -> int | None:
        if self.process.stdin is None or self.process.stdout is None:
            return None
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            return None
        parsed = json.loads(response)
        index = parsed.get("index")
        return int(index) if isinstance(index, int) else None

    def __call__(self, payload: dict[str, Any]) -> int | None:
        candidates = list(payload.get("candidates", []))
        index = self._query(payload)
        return index if index is not None and 0 <= index < len(candidates) else None

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.stderr_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--subset", choices=["train", "dev"], default="train")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases-per-task", type=int, default=20)
    parser.add_argument("--min-coverage", type=float, default=0.10)
    parser.add_argument("--min-supported-coverage", type=float, default=0.50)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-rescue-rate", type=float, default=0.10)
    parser.add_argument("--max-harm-rate", type=float, default=0.01)
    parser.add_argument("--max-reference-failure-rate", type=float, default=0.05)
    parser.add_argument("--min-schema-match-rate", type=float, default=0.95)
    parser.add_argument("--min-selector-candidates", type=int, default=1)
    parser.add_argument("--selector-python", type=Path, required=True)
    parser.add_argument("--selector-model", type=Path, required=True)
    parser.add_argument("--selector-device", default="cuda:0")
    parser.add_argument("--selector-script", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/appworld_harness_audit_30")
    )
    args = parser.parse_args()
    os.environ["APPWORLD_ROOT"] = str(args.appworld_root.resolve())

    from appworld import AppWorld, load_task_ids, update_root

    update_root(str(args.appworld_root.resolve()))
    all_task_ids = list(load_task_ids(args.subset))
    task_ids = all_task_ids[args.offset : args.offset + args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema_index = OpenAPISchemaIndex(
        args.appworld_root / "data" / "api_docs" / "openapi"
    )
    selector = SelectorWorker(
        args.selector_python,
        args.selector_script
        or Path(__file__).with_name("appworld_candidate_selector_worker.py"),
        args.selector_model,
        args.selector_device,
        args.output_dir / "selector_stderr.log",
    )
    harness = AppWorldCandidateHarness(
        selector,
        min_selector_candidates=args.min_selector_candidates,
    )
    records: list[dict[str, Any]] = []
    clean_cases = 0
    harmful_clean_changes = 0
    corrupt_cases = 0
    visible_support_cases = 0
    corrections = 0
    correct_corrections = 0
    supported_correct_corrections = 0
    reference_execution_failures = 0
    reference_calls = 0
    schema_matches = 0
    schema_misses = 0
    started = time.time()
    try:
        for task_index, task_id in enumerate(task_ids):
            world = AppWorld(
                task_id=task_id,
                experiment_name=f"rescuecredit_aw2_harness_{args.seed}_{task_index}",
                ground_truth_mode="full",
                raise_on_failure=False,
                random_seed=args.seed + task_index,
            )
            try:
                instruction = str(world.task.instruction)
                calls = list(getattr(world.task.ground_truth, "api_calls", []) or [])
                visible_receipts: dict[str, Any] | None = None
                task_cases = 0
                for call_index, call in enumerate(calls):
                    expected = _canonical(call)
                    arguments = dict(expected["arguments"])
                    public_schema = schema_index.schema_for(call)
                    required_fields = (
                        public_schema.get("required_fields", [])
                        if public_schema is not None
                        else None
                    )
                    if public_schema is None:
                        schema_misses += int(not _is_supervisor(call))
                    else:
                        schema_matches += int(not _is_supervisor(call))
                    if (
                        not _is_supervisor(call)
                        and arguments
                        and required_fields
                        and task_cases < args.max_cases_per_task
                    ):
                        clean_cases += 1
                        clean_action, clean_decision = harness.repair(
                            instruction,
                            visible_receipts,
                            copy.deepcopy(expected),
                            required_fields,
                            public_schema=public_schema,
                        )
                        clean_harm = clean_decision.changed and clean_action != expected
                        harmful_clean_changes += int(clean_harm)

                        for parameter in required_fields:
                            if task_cases >= args.max_cases_per_task:
                                break
                            proposal = copy.deepcopy(expected)
                            proposal["arguments"].pop(parameter, None)
                            expected_value = arguments[parameter]
                            _, supported_values = harness.candidates(
                                instruction, visible_receipts, parameter
                            )
                            supported = any(
                                type(value) is type(expected_value) and value == expected_value
                                for value in supported_values
                            )
                            corrupt_cases += 1
                            visible_support_cases += int(supported)
                            repaired, decision = harness.repair(
                                instruction,
                                visible_receipts,
                                proposal,
                                required_fields,
                                public_schema=public_schema,
                            )
                            changed = bool(decision.changed and repaired != proposal)
                            correct = changed and repaired == expected
                            corrections += int(changed)
                            correct_corrections += int(correct)
                            supported_correct_corrections += int(correct and supported)
                            records.append(
                                {
                                    "task_id_hash": _hash_task(str(task_id)),
                                    "call_index": call_index,
                                    "tool_hash": _tool_hash(call),
                                    "case": "missing_observed_argument",
                                    "parameter_hash": hashlib.sha256(parameter.encode()).hexdigest(),
                                    "visible_support": supported,
                                    "candidate_count": decision.candidate_count,
                                    "selected_by": decision.selected_by,
                                    "selected_sources": list(decision.selected_sources),
                                    "selected_origins": list(decision.selected_origins),
                                    "changed": changed,
                                    "correct_after": correct,
                                    "patch_id": decision.patch_id,
                                    "reference_scope": "offline_case_construction_and_scoring_only",
                                    "protected_values_exported": False,
                                }
                            )
                            task_cases += 1

                    output = str(world.execute(_render_rest_call(call)))
                    receipt = _receipt(output)
                    reference_calls += 1
                    reference_execution_failures += int(receipt.get("status") == "error")
                    visible_receipts = _merge_receipt(visible_receipts, receipt)
            finally:
                world.close()
    finally:
        selector.close()

    records_path = args.output_dir / "case_results.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    coverage = corrections / max(1, corrupt_cases)
    supported_coverage = supported_correct_corrections / max(1, visible_support_cases)
    precision = correct_corrections / max(1, corrections)
    rescue_rate = correct_corrections / max(1, corrupt_cases)
    harm_rate = harmful_clean_changes / max(1, clean_cases)
    reference_failure_rate = reference_execution_failures / max(1, reference_calls)
    schema_match_rate = schema_matches / max(1, schema_matches + schema_misses)
    metrics = {
        "tasks": len(task_ids),
        "clean_cases": clean_cases,
        "corrupt_cases": corrupt_cases,
        "visible_support_cases": visible_support_cases,
        "corrections": corrections,
        "correct_corrections": correct_corrections,
        "coverage": coverage,
        "supported_coverage": supported_coverage,
        "correction_precision": precision,
        "single_step_rescue_rate": rescue_rate,
        "harm_rate": harm_rate,
        "reference_execution_failures": reference_execution_failures,
        "reference_calls": reference_calls,
        "reference_execution_failure_rate": reference_failure_rate,
        "public_schema_matches": schema_matches,
        "public_schema_misses": schema_misses,
        "public_schema_match_rate": schema_match_rate,
        "reference_boundary": {
            "harness_inputs": [
                "task_instruction",
                "AppWorld_public_OpenAPI_schema",
                "previous_visible_receipts",
                "proposal",
            ],
            "reference_values": "offline case construction and scoring only",
            "required_fields": "AppWorld data/api_docs/openapi only",
            "output_contains_protected_values": False,
        },
        "audit_scope": "visible-candidate missing REST argument reconstruction",
        "min_selector_candidates": args.min_selector_candidates,
        "task_offset": args.offset,
        "wall_time_sec": time.time() - started,
    }
    passed = bool(
        coverage >= args.min_coverage
        and supported_coverage >= args.min_supported_coverage
        and precision >= args.min_precision
        and rescue_rate >= args.min_rescue_rate
        and harm_rate <= args.max_harm_rate
        and reference_failure_rate <= args.max_reference_failure_rate
        and schema_match_rate >= args.min_schema_match_rate
    )
    gate = {
        "passed": passed,
        "stage": "appworld_deployable_harness_audit_30",
        "thresholds": {
            "min_coverage": args.min_coverage,
            "min_supported_coverage": args.min_supported_coverage,
            "min_correction_precision": args.min_precision,
            "min_single_step_rescue_rate": args.min_rescue_rate,
            "max_harm_rate": args.max_harm_rate,
            "max_reference_execution_failure_rate": args.max_reference_failure_rate,
            "min_public_schema_match_rate": args.min_schema_match_rate,
        },
        "metrics": metrics,
        "authorizes_v2_training_smoke": passed,
        "next_step": (
            "implement AppWorld V2 training smoke"
            if passed
            else "improve AppWorld deployable harness before training"
        ),
    }
    write_json(args.output_dir / "harness_metrics.json", metrics)
    write_json(args.output_dir / "quality_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if passed else 1)


if __name__ == "__main__":
    main()
