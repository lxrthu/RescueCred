#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from rescuecredit.appworld_shadow_credit import json_object
from rescuecredit.azure_client import AzureOpenAIAdapter


def _decode_object(text: str) -> dict[str, Any] | None:
    """Decode the first complete JSON object without inventing an action."""

    parsed = json_object(text)
    if parsed is not None:
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


def _validated_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    parsed = _decode_object(text)
    if parsed is None:
        return None, "invalid_json"
    if parsed.get("stop") is True:
        return {"action": None, "stopped": True}, None
    tool = parsed.get("tool")
    arguments = parsed.get("arguments")
    if not isinstance(tool, str) or "__" not in tool or not isinstance(arguments, dict):
        return None, "invalid_action_shape"
    return {"action": {"tool": tool, "arguments": arguments}}, None


def _complete_with_transport_retries(
    client: AzureOpenAIAdapter, messages: list[dict[str, Any]]
) -> tuple[str | None, str | None]:
    last_error = "unknown"
    for attempt in range(4):
        try:
            return client.complete(messages, max_tokens=500, temperature=0.0), None
        except Exception as error:
            last_error = type(error).__name__
            if attempt < 3:
                time.sleep(2**attempt)
    return None, last_error


def _next_action(client: AzureOpenAIAdapter, request: dict[str, Any]) -> dict[str, Any]:
    visible = {
        "task_instruction": request.get("instruction", ""),
        "frozen_event_context": request.get("event_context", ""),
        "public_tool_schemas": request.get("tool_schemas", []),
        "visible_branch_history": request.get("history", []),
        "remaining_steps": request.get("remaining_steps", 0),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a deterministic AppWorld continuation policy. You receive no reference "
                "trajectory, expected action, hidden label, or evaluator output. Choose exactly one "
                "next public API call from visible information. Use api_docs APIs when documentation "
                "is insufficient. Call supervisor.complete_task only when the task is actually done. "
                "Never invent a hidden value. Return JSON only as "
                '{"tool":"app__api","arguments":{...}} or {"stop":true}.'
            ),
        },
        {
            "role": "user",
            "content": json.dumps(visible, ensure_ascii=False, sort_keys=True),
        },
    ]
    raw, transport_error = _complete_with_transport_retries(client, messages)
    if raw is None:
        return {"action": None, "error_type": transport_error}
    response, validation_error = _validated_response(raw)
    if response is not None:
        return response

    # One deterministic format-repair attempt. The repair prompt contains only the
    # model's own visible-output text and the public response schema. It never adds
    # reference actions, evaluator values, or a default action.
    repair_messages = messages + [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                f"Your previous response failed validation ({validation_error}). "
                "Return exactly one JSON object and no prose: "
                '{"tool":"app__api","arguments":{...}} or {"stop":true}. '
                "Preserve your intended action; repair only JSON syntax and schema shape."
            ),
        },
    ]
    repaired_raw, repair_transport_error = _complete_with_transport_retries(
        client, repair_messages
    )
    if repaired_raw is None:
        return {
            "action": None,
            "error_type": repair_transport_error,
            "format_repair_attempted": True,
        }
    repaired, repaired_error = _validated_response(repaired_raw)
    if repaired is None:
        return {
            "action": None,
            "error_type": repaired_error,
            "format_repair_attempted": True,
        }
    repaired["format_repair_attempted"] = True
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--device")
    parser.parse_args()
    client = AzureOpenAIAdapter()
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            response = _next_action(client, request)
        except Exception as error:
            response = {"action": None, "error_type": type(error).__name__}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
