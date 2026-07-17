#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from environments.appworld import normalize_function_tools
from rescuecredit.logging import write_json


def shape(value: Any, depth: int = 0) -> Any:
    """Describe protected objects without exporting task content."""

    if depth >= 3:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "length": len(value),
            "keys": sorted(map(str, value.keys())),
            "value_types": sorted({type(child).__name__ for child in value.values()}),
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "item_types": sorted({type(child).__name__ for child in value}),
            "first_item": shape(value[0], depth + 1) if value else None,
        }
    for method in ("model_dump", "dict", "to_dict"):
        candidate = getattr(value, method, None)
        if callable(candidate):
            return {
                "type": type(value).__name__,
                "serialized": shape(candidate(), depth + 1),
            }
    return {"type": type(value).__name__}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--subset", choices=["train", "dev"], default="train")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/appworld_contract_probe")
    )
    args = parser.parse_args()

    os.environ["APPWORLD_ROOT"] = str(args.appworld_root.resolve())
    try:
        import appworld
        from appworld import AppWorld, load_task_ids, update_root
    except ImportError as error:
        raise SystemExit(
            "AppWorld is not installed. Run scripts/cloud/setup_appworld_stage0.sh first."
        ) from error

    update_root(str(args.appworld_root.resolve()))
    task_ids = list(load_task_ids(args.subset))[: args.limit]
    if not task_ids:
        raise RuntimeError(f"AppWorld subset {args.subset!r} is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    started = time.time()
    for index, task_id in enumerate(task_ids):
        world = AppWorld(
            task_id=task_id,
            experiment_name=f"rescuecredit_contract_probe_{args.seed}_{index}",
            ground_truth_mode="full",
            raise_on_failure=False,
            random_seed=args.seed + index,
        )
        try:
            tools = normalize_function_tools(world.task.api_docs)
            state_id = world.save_state()
            if not state_id or not callable(getattr(world, "load_state", None)):
                raise RuntimeError("AppWorld checkpoint methods are unavailable")
            ground_truth = world.task.ground_truth
            api_calls = getattr(ground_truth, "api_calls", None)
            compiled_solution = getattr(ground_truth, "compiled_solution_module", None)
            records.append(
                {
                    "task_id_hash": hashlib.sha256(str(task_id).encode()).hexdigest(),
                    "function_tools": len(tools),
                    "tool_name_examples": [item["name"] for item in tools[:3]],
                    "api_calls_shape": shape(api_calls),
                    "compiled_solution_shape": shape(compiled_solution),
                    "save_load_api_callable": True,
                    "ground_truth_used_only_for_contract_shape": True,
                }
            )
        finally:
            world.close()

    gate = {
        "passed": bool(
            len(records) == len(task_ids)
            and all(record["function_tools"] > 0 for record in records)
            and all(record["save_load_api_callable"] for record in records)
            and all(record["api_calls_shape"]["type"] != "NoneType" for record in records)
        ),
        "stage": "appworld_contract_probe",
        "subset": args.subset,
        "num_tasks": len(records),
        "checks": {
            "public_function_schemas_present": all(
                record["function_tools"] > 0 for record in records
            ),
            "save_load_api_callable": all(
                record["save_load_api_callable"] for record in records
            ),
            "train_dev_offline_api_calls_present": all(
                record["api_calls_shape"]["type"] != "NoneType" for record in records
            ),
            "no_ground_truth_values_exported": True,
        },
        "appworld_version": getattr(appworld, "__version__", "unknown"),
        "records": records,
        "wall_time_sec": time.time() - started,
        "next_step": "implement a real state-changing rollback probe, then the 30-task audit",
        "authorizes_causal_shadow": False,
        "shadow_blocker": (
            "AW1 must first execute a real state-changing branch and prove "
            "before == restored != mutated for both DB and control state"
        ),
    }
    write_json(args.output_dir / "contract_probe.json", gate)
    write_json(
        args.output_dir / "gate.json",
        {
            key: gate[key]
            for key in (
                "passed",
                "stage",
                "checks",
                "authorizes_causal_shadow",
                "shadow_blocker",
                "next_step",
            )
        },
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
