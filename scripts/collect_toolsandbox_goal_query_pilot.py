#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from environments.toolsandbox import ToolSandboxRuntime
from rescuecredit.deltaguard_protocol import load_public_sources, visible_instruction
from rescuecredit.deltaguard_probe import public_context_digest
from rescuecredit.deltaguard_toolsandbox import (
    public_structure_digest,
    remap_frozen_action,
    replay_visible_prefix,
)
from rescuecredit.frozen_bank import file_sha256, write_jsonl
from rescuecredit.goal_directed_query import (
    build_goal_directed_queries,
    build_goal_query_certificate,
    public_receipt_row,
    query_structure,
)
from rescuecredit.logging import write_json
from scripts.freeze_toolsandbox_goal_query_pilot import PROTOCOL_STATUS


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.time()
    protocol = _load(args.protocol_lock)
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("invalid goal-query protocol")
    for path, digest in protocol.get("source_code_sha256", {}).items():
        source = Path(path)
        if not source.is_file() or file_sha256(source) != digest:
            raise ValueError(f"goal-query source drift: {path}")
    public_paths = [Path(row["path"]) for row in protocol["public_sources"]]
    for path, frozen in zip(
        public_paths, protocol["public_sources"], strict=True
    ):
        if not path.is_file() or file_sha256(path) != frozen["sha256"]:
            raise ValueError(f"goal-query public source drift: {path}")
    public_by_id = {
        str(row["event_id"]): row for row in load_public_sources(public_paths)
    }

    runtime = ToolSandboxRuntime()
    scenarios = runtime.scenarios()
    rows = []
    errors: dict[str, int] = {}
    query_calls = 0
    replayed_actions = 0
    for index, frozen in enumerate(protocol["source_events"], start=1):
        event_id = str(frozen["event_id"])
        public = public_by_id.get(event_id)
        if public is None:
            raise ValueError(f"missing frozen public event: {event_id}")
        try:
            history = public.get("treatment_visible_history")
            if not isinstance(history, list):
                raise ValueError("public event lacks visible history")
            scenario = scenarios[str(frozen["scenario_name"])]
            prefix, replacements, replay_audit = replay_visible_prefix(
                runtime=runtime,
                scenario=scenario,
                target_history=[row for row in history if isinstance(row, Mapping)],
            )
            action_a = remap_frozen_action(public["action_a"], replacements)
            action_b = remap_frozen_action(public["action_b"], replacements)
            if public_structure_digest(action_a) != frozen["action_structure_a"]:
                raise RuntimeError("goal-query A structure drift")
            if public_structure_digest(action_b) != frozen["action_structure_b"]:
                raise RuntimeError("goal-query B structure drift")
            schemas = runtime.tool_schemas(prefix)
            queries = build_goal_directed_queries(
                action_a=action_a,
                action_b=action_b,
                schemas=schemas,
                instruction=visible_instruction(
                    [row for row in history if isinstance(row, Mapping)]
                ),
            )
            chosen = queries[0] if queries else None
            if public_structure_digest(query_structure(queries[:1])) != frozen[
                "query_structure"
            ]:
                raise RuntimeError("goal-query plan structure drift")
            prefix_digest = public_context_digest(runtime, prefix)
            receipt_row = None
            if chosen is not None:
                receipt = runtime.execute(
                    runtime.snapshot(prefix),
                    {"tool": chosen.tool, "arguments": chosen.arguments},
                )
                receipt_row = public_receipt_row(receipt)
                query_calls += 1
            if public_context_digest(runtime, prefix) != prefix_digest:
                raise RuntimeError("goal query mutated the frozen prefix")
            certificate = build_goal_query_certificate(
                action_a=action_a,
                action_b=action_b,
                schemas=schemas,
                query=chosen,
                query_receipt=receipt_row,
            )
            replayed_actions += int(replay_audit["prefix_actions_replayed"])
            rows.append(
                {
                    "event_id": event_id,
                    "task_id_hash": str(frozen["task_id_hash"]),
                    "family": str(frozen["family"]),
                    "query_receipt": receipt_row,
                    "certificate": certificate,
                    "route_to_a": bool(certificate["route_to_a"]),
                    "schema_only_route_to_a": bool(
                        certificate["schema_only_route_to_a"]
                    ),
                    "query_incremental_route_to_a": bool(
                        certificate["query_incremental_route_to_a"]
                    ),
                    "prefix_unchanged": True,
                    "replay_audit": replay_audit,
                    "collection_error": None,
                }
            )
        except Exception as error:
            name = type(error).__name__
            errors[name] = errors.get(name, 0) + 1
            rows.append(
                {
                    "event_id": event_id,
                    "task_id_hash": str(frozen["task_id_hash"]),
                    "family": str(frozen["family"]),
                    "query_receipt": None,
                    "certificate": None,
                    "route_to_a": False,
                    "schema_only_route_to_a": False,
                    "query_incremental_route_to_a": False,
                    "prefix_unchanged": None,
                    "replay_audit": None,
                    "collection_error": {"type": name, "message": str(error)},
                }
            )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(protocol['source_events'])}",
                    "query_calls": query_calls,
                    "errors": sum(errors.values()),
                }
            ),
            flush=True,
        )

    args.output_dir.mkdir(parents=True)
    ledger_path = args.output_dir / "query_ledger.jsonl"
    write_jsonl(ledger_path, rows)
    manifest = {
        "status": "completed",
        "stage": "toolsandbox_goal_directed_query_pilot0_collection",
        "events": len(rows),
        "successful_events": sum(row["collection_error"] is None for row in rows),
        "collection_errors": errors,
        "query_calls": query_calls,
        "queries_per_event": query_calls / len(rows) if rows else 0.0,
        "a_branch_executions": 0,
        "b_branch_executions": 0,
        "prefix_actions_replayed": replayed_actions,
        "prefix_rollback_failures": 0,
        "labels_read": False,
        "official_evaluator_called": False,
        "hidden_state_exported": False,
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "query_ledger_sha256": file_sha256(ledger_path),
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "collection_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
