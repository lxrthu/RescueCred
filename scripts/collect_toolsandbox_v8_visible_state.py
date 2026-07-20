#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from environments.toolsandbox import (
    TOOL_SANDBOX_COMMIT,
    ToolSandboxRuntime,
    action_schema_complete,
    canonical_action,
)
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_protocol import current_toolsandbox_runtime_identity
from scripts.freeze_toolsandbox_v8_protocol import PROTOCOL_STATUS


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _action_from_visible_code(
    runtime: ToolSandboxRuntime, context: Any, code: str
) -> dict[str, Any]:
    tree = ast.parse(code)
    arguments: dict[str, Any] | None = None
    execution_name: str | None = None
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(statement.value, ast.Call):
            continue
        if target.id == "_rescuecredit_arguments":
            call = statement.value
            if not call.args:
                continue
            encoded = ast.literal_eval(call.args[0])
            decoded = json.loads(encoded)
            if isinstance(decoded, dict):
                arguments = decoded
        elif target.id == "_rescuecredit_result":
            execution_name = ast.unparse(statement.value.func)
    if arguments is None or execution_name is None:
        raise ValueError("V8 could not parse a frozen visible tool call")

    public_to_execution: dict[str, str] = {}
    for schema in runtime.tool_schemas(context):
        function = schema.get("function", {})
        public_name = function.get("name")
        if isinstance(public_name, str):
            public_to_execution[public_name] = context.get_execution_facing_tool_name(
                public_name
            )
    matches = [
        public_name
        for public_name, candidate in public_to_execution.items()
        if candidate == execution_name
    ]
    if len(matches) != 1:
        raise ValueError("V8 frozen visible tool name is unavailable or ambiguous")
    return {"tool": matches[0], "arguments": arguments}


def _reconstruct_prefix(
    runtime: ToolSandboxRuntime,
    scenario: Any,
    target_history: list[dict[str, Any]],
) -> tuple[Any, int]:
    prefix = runtime.prepare(scenario)
    replayed = 0
    while True:
        current = runtime.visible_history(prefix)
        if current == target_history:
            return prefix, replayed
        if len(current) >= len(target_history) or target_history[: len(current)] != current:
            raise RuntimeError("V8 frozen treatment history is not replayable from scenario")
        next_row = target_history[len(current)]
        if not isinstance(next_row, Mapping):
            raise RuntimeError("V8 frozen treatment history row is malformed")
        action = _action_from_visible_code(runtime, prefix, str(next_row.get("content", "")))
        receipt = runtime.execute(prefix, action)
        prefix = receipt.context
        replayed += 1
        updated = runtime.visible_history(prefix)
        if target_history[: len(updated)] != updated:
            raise RuntimeError("V8 deterministic prefix replay diverged from frozen history")


def _visible_state_summary(
    runtime: ToolSandboxRuntime, prefix: Any, action: Mapping[str, Any]
) -> dict[str, Any]:
    before_history = runtime.visible_history(prefix)
    schemas_before = runtime.tool_schemas(prefix)
    receipt = runtime.execute(runtime.snapshot(prefix), action)
    after_history = runtime.visible_history(receipt.context)
    if after_history[: len(before_history)] != before_history:
        raise RuntimeError("one-step probe rewrote visible history instead of appending")
    return {
        "receipt": {
            "action": receipt.action,
            "content": receipt.content,
            "exception": receipt.exception,
        },
        "appended_visible_history": after_history[len(before_history) :],
        "schemas_before": schemas_before,
        "schemas_after": runtime.tool_schemas(receipt.context),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--raw-events", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    protocol = _load(args.protocol_lock)
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("invalid V8 protocol")
    if file_sha256(args.raw_events) != protocol.get("raw_events_sha256"):
        raise ValueError("V8 raw event identity mismatch")
    if file_sha256(args.train_file) != protocol.get("train_file_sha256"):
        raise ValueError("V8 train event identity mismatch")
    if file_sha256(args.worker_script) != protocol.get("worker_script_sha256"):
        raise ValueError("V8 worker identity mismatch")
    if current_toolsandbox_runtime_identity(TOOL_SANDBOX_COMMIT) != protocol.get(
        "toolsandbox_runtime"
    ):
        raise ValueError("V8 ToolSandbox runtime identity mismatch")
    if not all(
        Path(path).is_file() and file_sha256(Path(path)) == digest
        for path, digest in protocol.get("source_sha256", {}).items()
    ):
        raise ValueError("V8 source identity changed")

    raw_by_id = {str(row["event_id"]): row for row in read_jsonl(args.raw_events)}
    train_rows = read_jsonl(args.train_file)
    expected = {str(row["event_id"]): raw_by_id[str(row["event_id"])] for row in train_rows}
    runtime = ToolSandboxRuntime()
    config = protocol["replay_config"]
    selected = runtime.select_scenarios(
        limit=int(config["limit"]),
        seed=int(config["seed"]),
        offset=int(config["scenario_offset"]),
        allow_distraction_tools=True,
    )
    selected_hashes = [
        hashlib.sha256(name.encode("utf-8")).hexdigest() for name, _ in selected
    ]
    if selected_hashes != protocol["scenario_identity"]["fresh_hashes"]:
        raise ValueError("V8 scenario identity mismatch")
    scenario_by_name = {name: scenario for name, scenario in selected}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    collected: dict[str, dict[str, Any]] = {}
    prefix_actions_replayed = 0
    ordered = sorted(
        expected.values(),
        key=lambda row: (
            str(row["scenario_name"]),
            int(row["reference_free_prefix_steps"]),
            int(row["candidate_rank"]),
        ),
    )
    for event_index, frozen in enumerate(ordered, start=1):
        scenario_name = str(frozen["scenario_name"])
        scenario = scenario_by_name.get(scenario_name)
        if scenario is None:
            raise RuntimeError(f"V8 frozen scenario is unavailable: {scenario_name}")
        target_history = frozen.get("treatment_visible_history")
        if not isinstance(target_history, list):
            raise RuntimeError("V8 frozen event lacks treatment-visible history")
        prefix, replayed = _reconstruct_prefix(runtime, scenario, target_history)
        prefix_actions_replayed += replayed
        schemas = runtime.tool_schemas(prefix)
        if schemas != frozen.get("treatment_public_tool_schemas"):
            raise RuntimeError("V8 reconstructed public schemas differ from frozen event")
        action_a = canonical_action(frozen["action_a"])
        action_b = canonical_action(frozen["action_b"])
        if not action_schema_complete(action_a, schemas) or not action_schema_complete(
            action_b, schemas
        ):
            raise RuntimeError("V8 frozen A/B is no longer schema-complete")
        event_id = str(frozen["event_id"])
        prefix_digest = runtime.context_digest(prefix)
        summary_a = _visible_state_summary(runtime, prefix, action_a)
        if runtime.context_digest(prefix) != prefix_digest:
            raise RuntimeError("V8 A probe mutated the frozen prefix")
        summary_b = _visible_state_summary(runtime, prefix, action_b)
        if runtime.context_digest(prefix) != prefix_digest:
            raise RuntimeError("V8 B probe mutated the frozen prefix")
        collected[event_id] = {
            "event_id": event_id,
            "task_id_hash": str(frozen["task_id_hash"]),
            "scenario_name": scenario_name,
            "reference_free_prefix_steps": int(
                frozen["reference_free_prefix_steps"]
            ),
            "candidate_rank": int(frozen["candidate_rank"]),
            "action_a": action_a,
            "action_b": action_b,
            "state_summary_a": summary_a,
            "state_summary_b": summary_b,
        }
        print(
            json.dumps(
                {
                    "progress": f"{event_index}/{len(ordered)}",
                    "collected": len(collected),
                }
            ),
            flush=True,
        )

    if set(collected) != set(expected):
        missing = sorted(set(expected) - set(collected))
        raise RuntimeError(f"V8 did not reproduce all frozen events: {missing[:5]}")
    rows = [collected[event_id] for event_id in sorted(collected)]
    event_path = args.output_dir / "visible_state_events.jsonl"
    write_jsonl(event_path, rows)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_v8_one_step_visible_state_collection",
        "events": len(rows),
        "tasks": len({row["task_id_hash"] for row in rows}),
        "live_worker_requests": 0,
        "prefix_actions_replayed": prefix_actions_replayed,
        "event_file_sha256": file_sha256(event_path),
        "raw_events_sha256": file_sha256(args.raw_events),
        "train_file_sha256": file_sha256(args.train_file),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "source_worker_identity": protocol["worker_identity"],
        "official_evaluator_called": False,
        "hidden_context_exported": False,
        "visible_outputs": [
            "first receipt",
            "appended agent-visible history",
            "public schemas before and after one action",
        ],
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "collection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
