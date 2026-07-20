#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from scripts.audit_toolsandbox_signal import Worker, _action_or_none, _worker_payload
from scripts.audit_toolsandbox_v44_candidates import (
    _candidate_payload,
    candidate_values_are_visible,
)
from scripts.freeze_toolsandbox_v8_protocol import PROTOCOL_STATUS


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--worker-model")
    parser.add_argument("--worker-device")
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
    worker_identity = {
        "provider": os.getenv("TOOLSANDBOX_LLM_PROVIDER", "deepseek"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://zhi-api.com/v1"),
        "thinking": os.getenv("DEEPSEEK_THINKING", "disabled"),
    }
    if worker_identity != protocol.get("worker_identity"):
        raise ValueError("V8 worker environment drifted from frozen V4.4 identity")
    if args.worker_model is not None and args.worker_model != worker_identity["model"]:
        raise ValueError("V8 worker CLI model disagrees with frozen worker identity")
    if not all(
        Path(path).is_file() and file_sha256(Path(path)) == digest
        for path, digest in protocol.get("source_sha256", {}).items()
    ):
        raise ValueError("V8 source identity changed")

    raw_by_id = {str(row["event_id"]): row for row in read_jsonl(args.raw_events)}
    train_rows = read_jsonl(args.train_file)
    expected = {str(row["event_id"]): raw_by_id[str(row["event_id"])] for row in train_rows}
    expected_by_key = {
        (
            str(row["scenario_name"]),
            int(row["reference_free_prefix_steps"]),
            int(row["candidate_rank"]),
        ): row
        for row in expected.values()
    }
    if len(expected_by_key) != len(expected):
        raise ValueError("V8 expected treatment keys are not unique")

    config = protocol["replay_config"]
    runtime = ToolSandboxRuntime()
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker = Worker(
        args.worker_python.resolve(),
        args.worker_script.resolve(),
        args.output_dir / "worker_stderr.log",
        args.worker_model,
        args.worker_device,
        float(config["worker_timeout_sec"]),
        str(config["harness_interface"]),
    )
    collected: dict[str, dict[str, Any]] = {}
    requests = 0
    try:
        for scenario_index, (scenario_name, scenario) in enumerate(selected, start=1):
            prefix = runtime.prepare(scenario)
            scenario_pairs = 0
            for prefix_step in range(
                min(int(config["horizon"]), int(config["event_search_steps"]))
            ):
                schemas = runtime.tool_schemas(prefix)
                requests += 1
                proposal_response = worker.request(
                    _worker_payload(
                        runtime,
                        prefix,
                        mode="propose",
                        remaining_steps=int(config["horizon"]) - prefix_step,
                    )
                )
                proposal_a = _action_or_none(proposal_response)
                if proposal_a is None:
                    break
                if not action_schema_complete(proposal_a, schemas):
                    raise RuntimeError("V8 proposal A is not schema-complete")
                probe = runtime.execute(runtime.snapshot(prefix), proposal_a)
                requests += 1
                candidate_response = worker.request(
                    _candidate_payload(
                        runtime,
                        prefix,
                        proposal_a,
                        int(config["horizon"]) - prefix_step,
                        int(config["candidate_count"]),
                    )
                )
                candidates = candidate_response.get("actions", [])
                if not isinstance(candidates, list):
                    candidates = []
                visible_history = runtime.visible_history(prefix)
                supported = [
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, Mapping)
                    and candidate_values_are_visible(
                        candidate, visible_history, proposal_a, schemas
                    )
                ]
                for candidate_rank, candidate_b in enumerate(supported):
                    if scenario_pairs >= int(config["max_pairs_per_scenario"]):
                        break
                    key = (scenario_name, prefix_step, candidate_rank)
                    if not action_schema_complete(candidate_b, schemas):
                        raise RuntimeError("V8 candidate B is not schema-complete")
                    frozen = expected_by_key.get(key)
                    if frozen is not None:
                        action_a = canonical_action(proposal_a)
                        action_b = canonical_action(candidate_b)
                        if action_a != canonical_action(frozen["action_a"]):
                            raise RuntimeError(f"V8 proposal drift at {key}")
                        if action_b != canonical_action(frozen["action_b"]):
                            raise RuntimeError(f"V8 candidate drift at {key}")
                        if visible_history != frozen["treatment_visible_history"]:
                            raise RuntimeError(f"V8 treatment history drift at {key}")
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
                            "reference_free_prefix_steps": prefix_step,
                            "candidate_rank": candidate_rank,
                            "action_a": action_a,
                            "action_b": action_b,
                            "state_summary_a": summary_a,
                            "state_summary_b": summary_b,
                        }
                    scenario_pairs += 1
                if scenario_pairs >= int(config["max_pairs_per_scenario"]):
                    break
                prefix = probe.context
            print(
                json.dumps(
                    {
                        "progress": f"{scenario_index}/{len(selected)}",
                        "collected": len(collected),
                        "expected": len(expected),
                    }
                ),
                flush=True,
            )
    finally:
        worker.close()

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
        "requests": requests,
        "event_file_sha256": file_sha256(event_path),
        "raw_events_sha256": file_sha256(args.raw_events),
        "train_file_sha256": file_sha256(args.train_file),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "worker_identity": worker_identity,
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
