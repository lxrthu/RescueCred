#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from environments.toolsandbox import (
    TOOL_SANDBOX_COMMIT,
    ToolSandboxRuntime,
    action_schema_complete,
)
from rescuecredit.frozen_bank import file_sha256, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_protocol import current_toolsandbox_runtime_identity
from scripts.audit_toolsandbox_signal import (
    Worker,
    _action_or_none,
    _paired_row,
    _run_branch,
    _worker_payload,
)
from scripts.freeze_toolsandbox_v44_candidate_protocol import PROTOCOL_STATUS


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar_values(value: Any) -> set[tuple[type, Any]]:
    if value is None or isinstance(value, (str, int, float, bool)):
        return {(type(value), value)}
    if isinstance(value, Mapping):
        result: set[tuple[type, Any]] = set()
        for child in value.values():
            result.update(_scalar_values(child))
        return result
    if isinstance(value, (list, tuple)):
        result = set()
        for child in value:
            result.update(_scalar_values(child))
        return result
    return set()


def _schema_literal_values(schemas: list[Mapping[str, Any]]) -> set[tuple[type, Any]]:
    literals: set[tuple[type, Any]] = set()

    def visit(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        for key in ("const", "default"):
            if key in value:
                literals.update(_scalar_values(value[key]))
        if isinstance(value.get("enum"), list):
            literals.update(_scalar_values(value["enum"]))
        for child in value.values():
            if isinstance(child, Mapping):
                visit(child)
            elif isinstance(child, list):
                for item in child:
                    visit(item)

    for schema in schemas:
        visit(schema)
    return literals


def candidate_values_are_visible(
    candidate: Mapping[str, Any],
    visible_history: list[Mapping[str, Any]],
    proposal_a: Mapping[str, Any],
    schemas: list[Mapping[str, Any]],
) -> bool:
    allowed = _scalar_values(visible_history) | _scalar_values(
        proposal_a.get("arguments", {})
    ) | _schema_literal_values(schemas)
    candidate_values = _scalar_values(candidate.get("arguments", {}))
    visible_strings = [
        value
        for value_type, value in _scalar_values(visible_history)
        if value_type is str
    ]
    return all(
        item in allowed
        or item[0] is str
        and bool(str(item[1]).strip())
        and any(str(item[1]) in source for source in visible_strings)
        for item in candidate_values
    )


def _validate_protocol(
    protocol_path: Path,
    args: argparse.Namespace,
    worker_script: Path,
) -> tuple[dict[str, Any], str]:
    protocol = _load(protocol_path)
    expected = {
        "status": PROTOCOL_STATUS,
        "role": args.role,
        "toolsandbox_commit": TOOL_SANDBOX_COMMIT,
        "seed": args.seed,
        "scenario_offset": args.scenario_offset,
        "limit": args.limit,
        "horizon": args.horizon,
        "event_search_steps": args.event_search_steps,
        "candidate_count": args.candidate_count,
        "max_pairs_per_scenario": args.max_pairs_per_scenario,
        "worker_timeout_sec": args.worker_timeout_sec,
        "harness_interface": args.harness_interface,
        "credit_mode": "lexicographic_v4",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(
                f"V4.4 protocol mismatch for {key}: {protocol.get(key)!r} != {value!r}"
            )
    source_hashes = protocol.get("source_sha256", {})
    root = Path(__file__).resolve().parents[1]
    if not source_hashes or not all(
        (root / relative).is_file()
        and file_sha256(root / relative) == expected_hash
        for relative, expected_hash in source_hashes.items()
    ):
        raise ValueError("V4.4 source identity changed after protocol freeze")
    if file_sha256(worker_script) != source_hashes.get(
        "scripts/toolsandbox_azure_worker.py"
    ):
        raise ValueError("V4.4 worker differs from the frozen protocol")
    artifact_paths = protocol.get("artifact_paths", {})
    artifact_hashes = protocol.get("artifact_identity", {})
    if not isinstance(artifact_paths, Mapping) or not isinstance(
        artifact_hashes, Mapping
    ):
        raise ValueError("V4.4 frozen artifact inventory is missing")
    for name, raw_path in artifact_paths.items():
        artifact = Path(str(raw_path))
        expected_hash = artifact_hashes.get(str(name) + "_sha256")
        if not artifact.is_file() or file_sha256(artifact) != expected_hash:
            raise ValueError(f"V4.4 frozen artifact identity changed: {name}")
    if protocol.get("toolsandbox_runtime") != current_toolsandbox_runtime_identity(
        TOOL_SANDBOX_COMMIT
    ):
        raise ValueError("ToolSandbox runtime identity changed")
    return protocol, file_sha256(protocol_path)


def _candidate_payload(
    runtime: ToolSandboxRuntime,
    context: Any,
    proposal_a: Mapping[str, Any],
    remaining_steps: int,
    candidate_count: int,
) -> dict[str, Any]:
    return {
        "mode": "diversify",
        "history": runtime.visible_history(context),
        "tool_schemas": runtime.tool_schemas(context),
        "remaining_steps": remaining_steps,
        "proposal_a": dict(proposal_a),
        "visible_receipt": None,
        "candidate_count": candidate_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("sanity", "full"), required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--worker-model")
    parser.add_argument("--worker-device")
    parser.add_argument("--worker-timeout-sec", type=float, default=600.0)
    parser.add_argument("--harness-interface", choices=("tool_id_v2",), default="tool_id_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-offset", type=int, default=85)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--event-search-steps", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, required=True)
    parser.add_argument("--max-pairs-per-scenario", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    worker_python = args.worker_python.resolve()
    worker_script = args.worker_script.resolve()
    if not worker_python.is_file() or not worker_script.is_file():
        raise FileNotFoundError("worker python or script is missing")
    protocol, protocol_sha256 = _validate_protocol(
        args.protocol_lock.resolve(), args, worker_script
    )
    runtime = ToolSandboxRuntime()
    selected = runtime.select_scenarios(
        limit=args.limit,
        seed=args.seed,
        offset=args.scenario_offset,
        allow_distraction_tools=True,
    )
    selected_hashes = [
        hashlib.sha256(name.encode("utf-8")).hexdigest() for name, _ in selected
    ]
    if selected_hashes != protocol["scenario_identity"]["fresh_hashes"]:
        raise ValueError("selected V4.4 scenarios differ from the frozen identity")

    rows: list[dict[str, Any]] = []
    requests = 0
    worker_failures = 0
    snapshot_checks = 0
    snapshot_mismatches = 0
    branch_prefix_checks = 0
    branch_prefix_mismatches = 0
    stats: Counter[str] = Counter()
    worker = Worker(
        worker_python,
        worker_script,
        output / "worker_stderr.log",
        args.worker_model,
        args.worker_device,
        args.worker_timeout_sec,
        args.harness_interface,
    )
    try:
        for scenario_index, (scenario_name, scenario) in enumerate(selected, start=1):
            prefix = runtime.prepare(scenario)
            task_hash = hashlib.sha256(scenario_name.encode("utf-8")).hexdigest()
            scenario_pairs = 0
            for prefix_step in range(min(args.horizon, args.event_search_steps)):
                schemas = runtime.tool_schemas(prefix)
                requests += 1
                stats["proposal_requests"] += 1
                try:
                    response = worker.request(
                        _worker_payload(
                            runtime,
                            prefix,
                            mode="propose",
                            remaining_steps=args.horizon - prefix_step,
                        )
                    )
                except Exception as error:
                    worker_failures += 1
                    stats["proposal_worker_" + type(error).__name__] += 1
                    break
                proposal_a = _action_or_none(response)
                if proposal_a is None:
                    stats["proposal_stop_or_invalid"] += 1
                    worker_failures += int(response.get("stopped") is not True)
                    break
                if not action_schema_complete(proposal_a, schemas):
                    raise RuntimeError("worker proposal A is not schema-complete")

                before = runtime.context_digest(prefix)
                restored = runtime.snapshot(prefix)
                snapshot_checks += 1
                if runtime.context_digest(restored) != before:
                    snapshot_mismatches += 1
                probe = runtime.execute(runtime.snapshot(prefix), proposal_a)

                requests += 1
                stats["candidate_requests"] += 1
                try:
                    candidate_response = worker.request(
                        _candidate_payload(
                            runtime,
                            prefix,
                            proposal_a,
                            args.horizon - prefix_step,
                            args.candidate_count,
                        )
                    )
                except Exception as error:
                    worker_failures += 1
                    stats["candidate_worker_" + type(error).__name__] += 1
                    prefix = probe.context
                    continue
                candidates = candidate_response.get("actions", [])
                if not isinstance(candidates, list):
                    candidates = []
                if not candidates:
                    worker_failures += int(candidate_response.get("error_type") is not None)
                    stats["candidate_empty"] += 1
                visible_history = runtime.visible_history(prefix)
                supported_candidates = [
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, Mapping)
                    and candidate_values_are_visible(
                        candidate, visible_history, proposal_a, schemas
                    )
                ]
                stats["candidate_actions"] += len(candidates)
                stats["candidate_rejected_unsupported_values"] += len(candidates) - len(
                    supported_candidates
                )
                if candidates and not supported_candidates:
                    worker_failures += 1
                for candidate_rank, candidate_b in enumerate(supported_candidates):
                    if scenario_pairs >= args.max_pairs_per_scenario:
                        break
                    if not isinstance(candidate_b, Mapping) or not action_schema_complete(
                        candidate_b, schemas
                    ):
                        raise RuntimeError("V4.4 candidate B is not schema-complete")
                    branch_a = _run_branch(
                        runtime, scenario, prefix, proposal_a, worker, args.horizon
                    )
                    branch_prefix_checks += 1
                    if runtime.context_digest(prefix) != before:
                        branch_prefix_mismatches += 1
                    branch_b = _run_branch(
                        runtime, scenario, prefix, candidate_b, worker, args.horizon
                    )
                    branch_prefix_checks += 1
                    if runtime.context_digest(prefix) != before:
                        branch_prefix_mismatches += 1
                    for branch in (branch_a, branch_b):
                        if branch.get("failure_reason") == "continuation_worker_failure":
                            worker_failures += 1
                    rows.append(
                        _paired_row(
                            "both_valid_candidate_pair",
                            scenario_name,
                            task_hash,
                            proposal_a,
                            candidate_b,
                            branch_a,
                            branch_b,
                            {
                                "reference_free_prefix_steps": prefix_step,
                                "candidate_rank": candidate_rank,
                                "candidate_set_size": len(supported_candidates),
                                "treatment_visible_history": visible_history,
                                "treatment_public_tool_schemas": schemas,
                                "both_actions_schema_complete": True,
                            },
                            "lexicographic_v4",
                            args.horizon,
                            f"prefix={prefix_step}:candidate={candidate_rank}",
                        )
                    )
                    scenario_pairs += 1
                if scenario_pairs >= args.max_pairs_per_scenario:
                    break
                prefix = probe.context
                stats["reference_free_prefix_advances"] += 1

            nonzero = sum(
                row.get("replay_valid") is True
                and row.get("decision") in {"rescue_preference", "reverse_preference"}
                for row in rows
            )
            print(
                json.dumps(
                    {
                        "progress": f"{scenario_index}/{len(selected)}",
                        "pairs": len(rows),
                        "nonzero": nonzero,
                        "scenario_pairs": scenario_pairs,
                    }
                ),
                flush=True,
            )
    finally:
        worker.close()

    valid = [row for row in rows if row.get("replay_valid") is True]
    nonzero_rows = [
        row
        for row in valid
        if row.get("decision") in {"rescue_preference", "reverse_preference"}
    ]
    decisions = Counter(str(row["decision"]) for row in valid)
    task_pairs = Counter(str(row["task_id_hash"]) for row in nonzero_rows)
    direction_tasks = {
        direction: len(
            {
                str(row["task_id_hash"])
                for row in nonzero_rows
                if row["decision"] == direction
            }
        )
        for direction in ("rescue_preference", "reverse_preference")
    }
    event_path = output / "candidate_events.jsonl"
    write_jsonl(event_path, rows)
    worker_failure_rate = worker_failures / max(1, requests + 2 * len(rows))
    summary = {
        "status": "completed",
        "stage": "toolsandbox_v44_candidate_diversity_audit",
        "role": args.role,
        "scenarios": len(selected),
        "pairs": len(rows),
        "valid_pairs": len(valid),
        "nonzero_pairs": len(nonzero_rows),
        "decisions": dict(sorted(decisions.items())),
        "direction_tasks": direction_tasks,
        "nonzero_tasks": len(task_pairs),
        "max_task_pairs": max(task_pairs.values(), default=0),
        "max_task_pair_share": max(task_pairs.values(), default=0)
        / max(1, len(nonzero_rows)),
        "requests": requests,
        "worker_failures": worker_failures,
        "worker_failure_rate": worker_failure_rate,
        "statistics": dict(sorted(stats.items())),
        "snapshot_audit": {
            "snapshot_checks": snapshot_checks,
            "snapshot_mismatches": snapshot_mismatches,
            "branch_prefix_checks": branch_prefix_checks,
            "branch_prefix_mismatches": branch_prefix_mismatches,
            "exact": snapshot_checks > 0
            and snapshot_mismatches == 0
            and branch_prefix_checks > 0
            and branch_prefix_mismatches == 0,
        },
        "protocol_validated": True,
        "protocol_lock_sha256": protocol_sha256,
        "selected_scenario_hashes": selected_hashes,
        "candidate_count": args.candidate_count,
        "max_pairs_per_scenario": args.max_pairs_per_scenario,
        "horizon": args.horizon,
        "harness_interface": args.harness_interface,
        "credit_mode": "lexicographic_v4",
        "event_file": str(event_path),
        "event_file_sha256": file_sha256(event_path),
        "worker_script_sha256": file_sha256(worker_script),
        "toolsandbox_runtime": current_toolsandbox_runtime_identity(
            TOOL_SANDBOX_COMMIT
        ),
        "reference_boundary": protocol["reference_boundary"],
        "wall_time_sec": time.time() - started,
    }
    write_json(output / "audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
