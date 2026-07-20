#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from environments.toolsandbox import ToolSandboxRuntime
from rescuecredit.deltaguard_certificate import build_delta_certificate
from rescuecredit.deltaguard_observers import (
    action_pair_family,
    build_observer_plan,
    plan_family,
    plan_structure_payload,
)
from rescuecredit.deltaguard_probe import run_paired_probe
from rescuecredit.deltaguard_protocol import (
    PROTOCOL_STATUS,
    load_public_sources,
    source_stream_digest,
    verify_protocol_source_identity,
    visible_instruction,
)
from rescuecredit.deltaguard_toolsandbox import (
    public_structure_digest,
    remap_frozen_action,
    replay_visible_prefix,
)
from rescuecredit.frozen_bank import file_sha256, write_jsonl
from rescuecredit.logging import write_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    protocol = _load(args.protocol_lock)
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("invalid DeltaGuard protocol")
    verify_protocol_source_identity(protocol)
    public_paths = [Path(row["path"]) for row in protocol["public_sources"]]
    frozen_public_sources = protocol["public_sources"]
    if len(public_paths) != len(frozen_public_sources):
        raise ValueError("DeltaGuard public source binding length mismatch")
    for path, frozen in zip(public_paths, frozen_public_sources):
        if not path.is_file() or file_sha256(path) != frozen["sha256"]:
            raise ValueError(f"DeltaGuard public source drift: {path}")
    for path, digest in protocol.get("source_sha256", {}).items():
        if not Path(path).is_file() or file_sha256(Path(path)) != digest:
            raise ValueError(f"DeltaGuard source drift: {path}")
    if source_stream_digest(protocol["source_events"]) != protocol["source_stream_sha256"]:
        raise ValueError("DeltaGuard frozen source stream drift")

    public_rows = load_public_sources(public_paths)
    public_by_id = {str(row["event_id"]): row for row in public_rows}
    runtime = ToolSandboxRuntime()
    scenarios = runtime.scenarios()
    source_ledger: list[dict[str, Any]] = []
    probe_ledger: list[dict[str, Any]] = []
    errors: dict[str, int] = {}
    observer_calls = 0
    branch_action_executions = 0
    prefix_actions_replayed = 0
    predicate_pairs = 0
    unknown_predicate_pairs = 0
    for index, frozen in enumerate(protocol["source_events"], start=1):
        event_id = str(frozen["event_id"])
        public = public_by_id.get(event_id)
        if public is None:
            raise ValueError(f"frozen event missing from public sources: {event_id}")
        source_row = {
            "event_id": event_id,
            "task_id_hash": str(frozen["task_id_hash"]),
            "scenario_name": str(frozen["scenario_name"]),
            "family": str(frozen["family"]),
            "eligible": bool(frozen["eligible"]),
            "selected": bool(frozen["selected"]),
            "action_hash_a_at_freeze": str(frozen["action_hash_a"]),
            "action_hash_b_at_freeze": str(frozen["action_hash_b"]),
        }
        source_ledger.append(source_row)
        if not source_row["selected"]:
            continue
        try:
            scenario = scenarios[str(frozen["scenario_name"])]
            history = public.get("treatment_visible_history")
            if not isinstance(history, list):
                raise ValueError("source event lacks visible treatment history")
            prefix, replacements, replay_audit = replay_visible_prefix(
                runtime=runtime,
                scenario=scenario,
                target_history=[row for row in history if isinstance(row, Mapping)],
            )
            action_a = remap_frozen_action(public["action_a"], replacements)
            action_b = remap_frozen_action(public["action_b"], replacements)
            if public_structure_digest(action_a) != frozen["action_structure_a"]:
                raise RuntimeError("A action structure changed during public replay")
            if public_structure_digest(action_b) != frozen["action_structure_b"]:
                raise RuntimeError("B action structure changed during public replay")
            schemas = runtime.tool_schemas(prefix)
            plan = build_observer_plan(
                action_a=action_a,
                action_b=action_b,
                schemas=schemas,
                instruction=visible_instruction(
                    [row for row in history if isinstance(row, Mapping)]
                ),
            )
            replayed_family = plan_family(plan) or action_pair_family(
                action_a, action_b, schemas
            )
            if replayed_family != frozen["family"]:
                raise RuntimeError("observer family changed during public replay")
            if len(plan) != int(frozen["plan_predicates"]):
                raise RuntimeError("observer predicate count changed during public replay")
            if public_structure_digest(plan_structure_payload(plan)) != frozen["plan_structure"]:
                raise RuntimeError("observer plan structure changed during public replay")
            evidence = run_paired_probe(
                runtime=runtime,
                prefix=prefix,
                action_a=action_a,
                action_b=action_b,
                plan=plan,
            )
            evidence["receipt_family"] = replayed_family
            certificate = build_delta_certificate(evidence)
            observer_calls += (
                len(evidence["pre_observations"])
                + len(evidence["branch_a"]["post_observations"])
                + len(evidence["branch_b"]["post_observations"])
            )
            branch_action_executions += 2
            prefix_actions_replayed += int(replay_audit["prefix_actions_replayed"])
            predicate_pairs += len(certificate["predicates"])
            unknown_predicate_pairs += sum(
                row["delta_a"] == "unknown" or row["delta_b"] == "unknown"
                for row in certificate["predicates"]
            )
            contract_enabled = False
            contract_result = {
                "contract_applied": False,
                "contract_valid": None,
                "contract_errors": ["contract ablation disabled until a pre-observation lock is implemented"],
            }
            contract_score = float(certificate["reverse_score"])
            probe_ledger.append(
                {
                    "event_id": event_id,
                    "task_id_hash": source_row["task_id_hash"],
                    "family": source_row["family"],
                    "evidence": evidence,
                    "certificate": certificate,
                    "reverse_score": float(certificate["reverse_score"]),
                    "route_to_a": bool(certificate["route_to_a"]),
                    "contract_enabled": contract_enabled,
                    "contract_reverse_score": contract_score,
                    "contract_route_to_a": contract_score == 1.0,
                    "contract_audit": contract_result,
                    "replay_audit": replay_audit,
                    "collection_error": None,
                }
            )
        except Exception as error:
            name = type(error).__name__
            errors[name] = errors.get(name, 0) + 1
            probe_ledger.append(
                {
                    "event_id": event_id,
                    "task_id_hash": source_row["task_id_hash"],
                    "family": source_row["family"],
                    "evidence": None,
                    "certificate": None,
                    "reverse_score": 0.5,
                    "route_to_a": False,
                    "contract_enabled": False,
                    "contract_reverse_score": 0.5,
                    "contract_route_to_a": False,
                    "contract_audit": {
                        "contract_applied": False,
                        "contract_valid": False,
                        "contract_errors": ["collection_error"],
                    },
                    "replay_audit": None,
                    "collection_error": {"type": name, "message": str(error)},
                }
            )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(protocol['source_events'])}",
                    "probes": len(probe_ledger),
                    "errors": sum(errors.values()),
                }
            ),
            flush=True,
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "source_ledger.jsonl"
    probe_path = output / "probe_ledger.jsonl"
    write_jsonl(source_path, source_ledger)
    write_jsonl(probe_path, probe_ledger)
    manifest = {
        "status": "completed",
        "stage": "toolsandbox_deltaguard_public_paired_collection",
        "source_events": len(source_ledger),
        "probe_events": len(probe_ledger),
        "probe_rate": len(probe_ledger) / len(source_ledger),
        "successful_probes": sum(row["collection_error"] is None for row in probe_ledger),
        "collection_errors": errors,
        "observer_calls": observer_calls,
        "branch_action_executions": branch_action_executions,
        "prefix_actions_replayed": prefix_actions_replayed,
        "predicate_pairs": predicate_pairs,
        "unknown_predicate_pairs": unknown_predicate_pairs,
        "unknown_rate": unknown_predicate_pairs / predicate_pairs if predicate_pairs else 0.0,
        "rollback_failures": 0,
        "observer_calls_per_successful_probe": (
            observer_calls
            / max(1, sum(row["collection_error"] is None for row in probe_ledger))
        ),
        "source_ledger_sha256": file_sha256(source_path),
        "probe_ledger_sha256": file_sha256(probe_path),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "contract_ablation_enabled": False,
        "labels_read": False,
        "official_evaluator_called": False,
        "hidden_state_exported": False,
        "wall_time_sec": time.time() - started,
    }
    write_json(output / "collection_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
