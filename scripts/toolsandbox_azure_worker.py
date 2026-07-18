#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

from rescuecredit.appworld_shadow_credit import json_object
from rescuecredit.azure_client import AzureOpenAIAdapter


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


def _validate(
    parsed: Optional[Mapping[str, Any]], schemas: List[Mapping[str, Any]], allow_abstain: bool
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(parsed, Mapping):
        return None, "invalid_json"
    if parsed.get("stop") is True:
        return {"stopped": True, "action": None}, None
    if allow_abstain and parsed.get("abstain") is True:
        return {"abstained": True, "action": None}, None
    tool = parsed.get("tool")
    arguments = parsed.get("arguments")
    if tool not in _tool_names(schemas):
        return None, "unknown_tool"
    if not isinstance(arguments, dict):
        return None, "arguments_not_object"
    return {"action": {"tool": tool, "arguments": arguments}}, None


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


def _request(client: AzureOpenAIAdapter, request: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(request.get("mode", "continue"))
    schemas = request.get("tool_schemas", [])
    if not isinstance(schemas, list):
        return {"action": None, "error_type": "schemas_not_list"}
    allow_abstain = mode == "repair"
    visible = {
        "mode": mode,
        "task_messages": request.get("history", []),
        "public_tool_schemas": schemas,
        "proposal_a": request.get("proposal_a"),
        "visible_receipt": request.get("visible_receipt"),
        "remaining_steps": request.get("remaining_steps", 0),
    }
    if mode == "repair":
        task = (
            "Audit proposal A after its visible execution error. If visible evidence and the "
            "public schema support exactly one safe correction, return it. Otherwise abstain. "
            "Do not invent values and do not use hidden task labels."
        )
        output = '{"tool":"name","arguments":{...}} or {"abstain":true}'
    else:
        task = (
            "Choose exactly one next tool call that advances the visible user request. "
            "Respect public schemas and visible tool receipts. Do not invent unavailable values."
        )
        output = '{"tool":"name","arguments":{...}} or {"stop":true}'
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
    result, validation_error = _validate(_decode(raw), schemas, allow_abstain)
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
    result, validation_error = _validate(_decode(repaired_raw), schemas, allow_abstain)
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
    parser.parse_args()
    provider = os.getenv("TOOLSANDBOX_LLM_PROVIDER", "azure")
    client = AzureOpenAIAdapter(provider=provider)
    for raw in sys.stdin:
        try:
            response = _request(client, json.loads(raw))
        except Exception as error:
            response = {"action": None, "error_type": type(error).__name__}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
