#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# The audit intentionally launches this worker from an isolated temporary cwd
# with a strict environment allowlist.  Resolve project code from the worker's
# own immutable location instead of inheriting PYTHONPATH from the parent.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rescuecredit.appworld_shadow_credit import json_object  # noqa: E402
from rescuecredit.azure_client import AzureOpenAIAdapter  # noqa: E402


HARNESS_INTERFACES = ("tool_name_v1", "tool_id_v2")


def _decode(text: str) -> Optional[Dict[str, Any]]:
    parsed = json_object(text)
    if isinstance(parsed, dict):
        return parsed
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _tool_names(schemas: List[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for entry in schemas:
        function = entry.get("function", entry)
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            names.add(str(function["name"]))
    return names


def _argument_schema_error(
    tool: str, arguments: Mapping[str, Any], schemas: List[Mapping[str, Any]]
) -> Optional[str]:
    for entry in schemas:
        function = entry.get("function", entry)
        if not isinstance(function, Mapping) or function.get("name") != tool:
            continue
        parameters = function.get("parameters", {})
        if not isinstance(parameters, Mapping):
            return "parameters_schema_invalid"
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            return "parameters_schema_invalid"
        if not set(required).issubset(arguments):
            return "missing_required_arguments"
        if not set(arguments).issubset(properties):
            return "unknown_arguments"
        return None
    return "unknown_tool"


def _indexed_tool_catalog(
    schemas: List[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    public_functions: List[Dict[str, Any]] = []
    for entry in schemas:
        function = entry.get("function", entry)
        if not isinstance(function, Mapping) or not isinstance(
            function.get("name"), str
        ):
            continue
        public_functions.append(
            {
                "name": str(function["name"]),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
        )
    public_functions.sort(key=lambda item: item["name"])
    catalog: List[Dict[str, Any]] = []
    mapping: Dict[str, str] = {}
    for index, function in enumerate(public_functions):
        tool_id = f"T{index:04d}"
        mapping[tool_id] = function["name"]
        catalog.append({"tool_id": tool_id, **function})
    return catalog, mapping


def _validate(
    parsed: Optional[Mapping[str, Any]],
    schemas: List[Mapping[str, Any]],
    allow_abstain: bool,
    harness_interface: str = "tool_name_v1",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(parsed, Mapping):
        return None, "invalid_json"
    if parsed.get("stop") is True:
        return {"stopped": True, "action": None}, None
    if allow_abstain and parsed.get("abstain") is True:
        return {"abstained": True, "action": None}, None
    selected_tool_id = None
    if harness_interface == "tool_id_v2":
        selected_tool_id = parsed.get("tool_id")
        _, tool_mapping = _indexed_tool_catalog(schemas)
        if not isinstance(selected_tool_id, str):
            return None, "tool_id_missing"
        tool = tool_mapping.get(selected_tool_id)
        if tool is None:
            return None, "unknown_tool_id"
    elif harness_interface == "tool_name_v1":
        tool = parsed.get("tool")
    else:
        return None, "unsupported_harness_interface"
    arguments = parsed.get("arguments")
    if tool not in _tool_names(schemas):
        return None, "unknown_tool"
    if not isinstance(arguments, dict):
        return None, "arguments_not_object"
    schema_error = _argument_schema_error(str(tool), arguments, schemas)
    if schema_error is not None:
        return None, schema_error
    result = {"action": {"tool": tool, "arguments": arguments}}
    if selected_tool_id is not None:
        result["selected_tool_id"] = selected_tool_id
        result["mapped_from_public_catalog"] = True
    return result, None


def _complete(
    client: AzureOpenAIAdapter, messages: List[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str]]:
    last_error = "unknown"
    for attempt in range(4):
        try:
            return client.complete(messages, max_tokens=700, temperature=0.0), None
        except Exception as error:
            last_error = type(error).__name__
            if attempt < 3:
                time.sleep(2**attempt)
    return None, last_error


def _request(
    client: AzureOpenAIAdapter,
    request: Dict[str, Any],
    harness_interface: str = "tool_name_v1",
) -> Dict[str, Any]:
    mode = str(request.get("mode", "continue"))
    schemas = request.get("tool_schemas", [])
    if not isinstance(schemas, list):
        return {"action": None, "error_type": "schemas_not_list"}
    allow_abstain = mode == "repair"
    visible: Dict[str, Any] = {
        "mode": mode,
        "task_messages": request.get("history", []),
        "proposal_a": request.get("proposal_a"),
        "visible_receipt": request.get("visible_receipt"),
        "remaining_steps": request.get("remaining_steps", 0),
    }
    if harness_interface == "tool_id_v2":
        visible["public_tool_catalog"], _ = _indexed_tool_catalog(schemas)
        action_output = '{"tool_id":"T0000","arguments":{...}}'
    else:
        visible["public_tool_schemas"] = schemas
        action_output = '{"tool":"name","arguments":{...}}'
    if mode == "repair":
        task = (
            "Audit proposal A after its visible execution error. If visible evidence and the "
            "public schema support exactly one safe correction, return it. Otherwise abstain. "
            "Do not invent values and do not use hidden task labels."
        )
        output = action_output + ' or {"abstain":true}'
    else:
        task = (
            "Choose exactly one next tool call that advances the visible user request. "
            "Respect public schemas and visible tool receipts. Do not invent unavailable values."
        )
        output = action_output + ' or {"stop":true}'
    messages = [
        {
            "role": "system",
            "content": (
                "You are a deterministic ToolSandbox policy and deployable Harness. "
                "You never receive milestone DAGs, minefields, reference actions, evaluator "
                "scores, or hidden database values. "
                + task
                + " Return JSON only as "
                + output
                + "."
            ),
        },
        {"role": "user", "content": json.dumps(visible, ensure_ascii=False, sort_keys=True)},
    ]
    raw, transport_error = _complete(client, messages)
    if raw is None:
        return {"action": None, "error_type": transport_error}
    result, validation_error = _validate(
        _decode(raw), schemas, allow_abstain, harness_interface
    )
    if result is not None:
        return result
    repaired_raw, repair_error = _complete(
        client,
        messages
        + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "Your output failed validation ("
                    + str(validation_error)
                    + "). Repair only JSON syntax/schema and return one allowed object."
                ),
            },
        ],
    )
    if repaired_raw is None:
        return {"action": None, "error_type": repair_error, "format_repair_attempted": True}
    result, validation_error = _validate(
        _decode(repaired_raw), schemas, allow_abstain, harness_interface
    )
    if result is None:
        return {
            "action": None,
            "error_type": validation_error,
            "format_repair_attempted": True,
        }
    result["format_repair_attempted"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--device")
    parser.add_argument(
        "--harness-interface", choices=HARNESS_INTERFACES, default="tool_name_v1"
    )
    args = parser.parse_args()
    provider = os.getenv("TOOLSANDBOX_LLM_PROVIDER", "azure")
    client = AzureOpenAIAdapter(provider=provider)
    for raw in sys.stdin:
        try:
            response = _request(client, json.loads(raw), args.harness_interface)
        except Exception as error:
            response = {"action": None, "error_type": type(error).__name__}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
