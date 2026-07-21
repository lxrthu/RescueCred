#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from rescuecredit.deltaguard_goal_contract import (
    infer_action_family,
    infer_goal_family,
)
from rescuecredit.deltaguard_protocol import (
    load_public_sources,
    visible_instruction,
)
from rescuecredit.deltaguard_toolsandbox import public_structure_digest
from rescuecredit.frozen_bank import file_sha256
from rescuecredit.goal_directed_query import (
    QUERY_VERSION,
    build_goal_directed_queries,
    query_structure,
    validate_action_schema,
)
from rescuecredit.logging import write_json


PROTOCOL_STATUS = "frozen_before_goal_query_pilot_collection"
SOURCE_FILES = (
    "rescuecredit/goal_directed_query.py",
    "rescuecredit/deltaguard_observers.py",
    "rescuecredit/deltaguard_protocol.py",
    "rescuecredit/deltaguard_toolsandbox.py",
    "scripts/freeze_toolsandbox_goal_query_pilot.py",
    "scripts/collect_toolsandbox_goal_query_pilot.py",
    "scripts/evaluate_toolsandbox_goal_query_pilot.py",
)


def _family(
    instruction: str,
    action_a: Mapping[str, Any],
    action_b: Mapping[str, Any],
    schemas: list[Mapping[str, Any]],
) -> str:
    return str(
        infer_goal_family(instruction)
        or infer_action_family(action_b, schemas)
        or infer_action_family(action_a, schemas)
        or "other"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-events", type=Path, nargs="+", required=True)
    parser.add_argument("--public-bank-manifest", type=Path, required=True)
    parser.add_argument("--target-events", type=int, default=30)
    parser.add_argument("--minimum-events", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.minimum_events <= 0 or args.target_events < args.minimum_events:
        raise ValueError("invalid goal-query pilot event counts")
    missing = [path for path in SOURCE_FILES if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing)

    manifest = json.loads(args.public_bank_manifest.read_text(encoding="utf-8"))
    public_hashes = [file_sha256(path) for path in args.public_events]
    if manifest.get("public_bank_sha256") not in public_hashes:
        raise ValueError("public-bank manifest does not bind supplied events")
    if manifest.get("protected_fields_exported") != []:
        raise ValueError("public bank contains protected fields")
    candidates = []
    exclusions: Counter[str] = Counter()
    for row in load_public_sources(args.public_events):
        history = row.get("treatment_visible_history")
        schemas = row.get("treatment_public_tool_schemas")
        action_a = row.get("action_a")
        action_b = row.get("action_b")
        if not (
            isinstance(history, list)
            and isinstance(schemas, list)
            and isinstance(action_a, Mapping)
            and isinstance(action_b, Mapping)
        ):
            exclusions["malformed_public_event"] += 1
            continue
        public_schemas = [schema for schema in schemas if isinstance(schema, Mapping)]
        instruction = visible_instruction(
            [item for item in history if isinstance(item, Mapping)]
        )
        queries = build_goal_directed_queries(
            action_a=action_a,
            action_b=action_b,
            schemas=public_schemas,
            instruction=instruction,
        )
        schema_a = validate_action_schema(action_a, public_schemas)
        schema_b = validate_action_schema(action_b, public_schemas)
        schema_witness_possible = bool(schema_a["valid"] and not schema_b["valid"])
        if not queries and not schema_witness_possible:
            exclusions["no_pre_observation_witness_channel"] += 1
            continue
        candidates.append(
            {
                "event_id": str(row["event_id"]),
                "task_id_hash": str(row.get("task_id_hash", "")),
                "scenario_name": str(row.get("scenario_name", "")),
                "family": _family(
                    instruction, action_a, action_b, public_schemas
                ),
                "action_structure_a": public_structure_digest(action_a),
                "action_structure_b": public_structure_digest(action_b),
                "query_structure": public_structure_digest(
                    query_structure(queries[:1])
                ),
                "query_candidates": len(queries),
                "frozen_query": queries[0].to_dict() if queries else None,
                "schema_witness_possible": schema_witness_possible,
            }
        )
    def order_key(row: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            f"GoalQuery/Pilot0/{args.seed}/{row['event_id']}".encode("utf-8")
        ).hexdigest()

    query_candidates = sorted(
        (row for row in candidates if int(row["query_candidates"]) > 0),
        key=order_key,
    )
    schema_only_candidates = sorted(
        (row for row in candidates if int(row["query_candidates"]) == 0),
        key=order_key,
    )
    selected = query_candidates[: args.target_events]
    selected.extend(
        schema_only_candidates[: max(0, args.target_events - len(selected))]
    )
    if len(selected) < args.minimum_events:
        raise RuntimeError(
            {
                "eligible_events": len(candidates),
                "minimum_events": args.minimum_events,
                "exclusions": dict(exclusions),
            }
        )
    family_counts = Counter(str(row["family"]) for row in selected)
    protocol = {
        "status": PROTOCOL_STATUS,
        "stage": "toolsandbox_goal_directed_query_pilot0_protocol",
        "query_version": QUERY_VERSION,
        "seed": args.seed,
        "target_events": args.target_events,
        "minimum_events": args.minimum_events,
        "source_events": selected,
        "source_event_ids_sha256": hashlib.sha256(
            json.dumps(
                [row["event_id"] for row in selected], separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "eligible_events": len(candidates),
        "eligible_query_events": len(query_candidates),
        "eligible_schema_only_events": len(schema_only_candidates),
        "exclusions": dict(exclusions),
        "family_counts": dict(family_counts),
        "public_sources": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in args.public_events
        ],
        "public_bank_manifest": str(args.public_bank_manifest),
        "public_bank_manifest_sha256": file_sha256(args.public_bank_manifest),
        "raw_label_source_sha256_sealed_in_public_manifest": manifest.get(
            "raw_source_sha256"
        ),
        "source_code_sha256": {
            path: file_sha256(Path(path)) for path in SOURCE_FILES
        },
        "labels_available_to_freezer": False,
        "labels_read": False,
        "selection_rule": (
            "applicability-conditioned query-channel events first, then schema-only "
            "events; seeded public hash order within each stratum"
        ),
        "collection_boundary": (
            "one read-only query on the replayed public prefix; no A/B branch execution"
        ),
        "claim_boundary": (
            "applicability-conditioned feasibility pilot; not a deployment-rate, "
            "formal-risk, or paper-facing confirmation result"
        ),
        "gate": {
            "min_events": args.minimum_events,
            "min_rescue_events": 3,
            "min_reverse_events": 3,
            "max_empirical_rescue_drop": 0.02,
            "min_reverse_recall": 0.20,
            "min_query_incremental_reverse_hits": 1,
            "max_queries_per_event": 1.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
