#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any

from rescuecredit.azure_client import AzureOpenAIAdapter


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group())
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _select(client: AzureOpenAIAdapter, request: dict[str, Any]) -> dict[str, Any]:
    candidates = list(request.get("candidates", []))
    sources = list(request.get("candidate_sources", []))
    origins = list(request.get("candidate_origins", []))
    public_schema = request.get("public_schema", {})
    candidate_rows = [
        {
            "index": index,
            "value": value,
            "visible_source": sources[index] if index < len(sources) else [],
            "visible_provenance": origins[index] if index < len(origins) else [],
        }
        for index, value in enumerate(candidates)
    ]
    payload = {
        "task_instruction": request.get("instruction", ""),
        "tool": request.get("tool", ""),
        "missing_argument": request.get("parameter", ""),
        "public_openapi_schema": public_schema,
        "visible_candidates": candidate_rows,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a conservative reference-free tool-argument validator. "
                "You never see or infer a hidden reference action. Select a candidate only when "
                "the task instruction, public OpenAPI schema, and visible-source label make it "
                "unambiguous. Candidate provenance paths identify the visible receipt field or "
                "instruction span that produced each value. Otherwise abstain. Never invent or "
                "transform a value. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Choose the exact visible candidate for the missing argument. Return exactly "
                '{"index": <integer or null>, "confidence": <number 0 to 1>}. '
                "Use null whenever more than one candidate is plausible or none is explicitly supported.\n"
                + json.dumps(payload, ensure_ascii=False, sort_keys=True)
            ),
        },
    ]
    last_error = "unknown"
    for attempt in range(4):
        try:
            parsed = _json_object(
                client.complete(messages, max_tokens=80, temperature=0.0)
            )
            if parsed is None:
                return {"index": None, "error_type": "invalid_json"}
            index = parsed.get("index")
            confidence = parsed.get("confidence", 0.0)
            if index is None:
                return {"index": None, "confidence": float(confidence or 0.0)}
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                return {"index": None, "error_type": "invalid_index"}
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                return {"index": None, "error_type": "invalid_confidence"}
            return {
                "index": index if confidence >= 0.90 else None,
                "confidence": confidence,
                "strategy": "azure_gpt4o_visible_candidate",
            }
        except Exception as error:
            last_error = type(error).__name__
            if attempt < 3:
                time.sleep(2**attempt)
    return {"index": None, "error_type": last_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    # Kept for the common worker interface. Azure deployment comes from .env.
    parser.add_argument("--model")
    parser.add_argument("--device")
    args = parser.parse_args()
    del args
    client = AzureOpenAIAdapter()
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            response = (
                _select(client, request)
                if request.get("candidates")
                else {"index": None}
            )
        except Exception as error:
            response = {"index": None, "error_type": type(error).__name__}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
