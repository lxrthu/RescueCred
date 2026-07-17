#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from rescuecredit.logging import write_json


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _save_digest(world: Any, label: str) -> tuple[str, str]:
    state_id = f"aw1_{label}_{uuid.uuid4().hex}"
    world.save_state(state_id)
    checkpoint = Path(world.output_checkpoints_directory) / state_id
    if not checkpoint.is_dir():
        raise RuntimeError(f"checkpoint was not materialized: {checkpoint}")
    return state_id, _hash_path(checkpoint)


def _control_state(world: Any) -> dict[str, Any]:
    tracker = getattr(getattr(world, "requester", None), "request_tracker", None)
    if tracker is None or not hasattr(tracker, "__dict__"):
        raise RuntimeError("AppWorld request tracker state is unavailable")
    numpy_random = None
    try:
        import numpy as np

        numpy_random = copy.deepcopy(np.random.get_state())
    except ImportError:
        pass
    return {
        "environment_io": copy.deepcopy(getattr(world, "environment_io", [])),
        "num_interactions": int(getattr(world, "num_interactions", 0)),
        "num_sub_interactions": int(getattr(world, "num_sub_interactions", 0)),
        "request_tracker": copy.deepcopy(tracker.__dict__),
        "python_random": random.getstate(),
        "numpy_random": numpy_random,
    }


def _control_digest(state: dict[str, Any]) -> str:
    return hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()


def _restore_control_state(world: Any, state: dict[str, Any]) -> None:
    world.environment_io = copy.deepcopy(state["environment_io"])
    world.num_interactions = state["num_interactions"]
    world.num_sub_interactions = state["num_sub_interactions"]
    tracker = getattr(getattr(world, "requester", None), "request_tracker", None)
    if tracker is None or not hasattr(tracker, "__dict__"):
        raise RuntimeError("AppWorld request tracker cannot be restored")
    tracker.__dict__.clear()
    tracker.__dict__.update(copy.deepcopy(state["request_tracker"]))
    random.setstate(state["python_random"])
    if state["numpy_random"] is not None:
        import numpy as np

        np.random.set_state(state["numpy_random"])


def _compat_load_state(world: Any, state_id: str) -> None:
    """Load DB state and repair AppWorld 0.1.3's cleared time freezer."""

    world.load_state(state_id)
    registry = getattr(type(world), "id_to_time_freezer", {})
    freezer_id = getattr(world, "time_freezer_id", None)
    if freezer_id not in registry:
        set_datetime = getattr(world, "_set_datetime", None)
        if not callable(set_datetime):
            raise RuntimeError("load_state cleared time control and it cannot be restored")
        set_datetime()


def _render_rest_call(call: dict[str, Any]) -> tuple[str, str]:
    """Replay AppWorld's native REST-form record without exporting values."""

    method = str(call.get("method", "")).lower()
    if method not in {"get", "post", "put", "patch", "delete"}:
        raise ValueError("unsupported AppWorld HTTP method")
    url = str(call.get("url", ""))
    if not url.startswith("/"):
        raise ValueError("AppWorld API-call URL must be a local absolute path")
    if url.startswith("/supervisor/"):
        raise PermissionError("supervisor calls are not application-mutation witnesses")
    arguments = call.get("data", {})
    if not isinstance(arguments, dict):
        raise TypeError("AppWorld API-call data must be an object")
    encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, allow_nan=False)
    code = (
        "import json\n"
        f"print(requester.{method}({url!r}, data=json.loads({encoded!r})))"
    )
    call_fingerprint = hashlib.sha256(f"{method}:{url}".encode()).hexdigest()
    return code, call_fingerprint


def _probe_task(AppWorld: Any, task_id: str, seed: int, index: int) -> dict[str, Any]:
    world = AppWorld(
        task_id=task_id,
        experiment_name=f"rescuecredit_aw1_rollback_{seed}_{index}",
        ground_truth_mode="full",
        raise_on_failure=False,
        random_seed=seed + index,
    )
    loaded_state = False
    try:
        before_id, before_db = _save_digest(world, "before")
        before_control = _control_state(world)
        before_control_hash = _control_digest(before_control)
        calls = list(getattr(world.task.ground_truth, "api_calls", []) or [])
        mutation_call_index = None
        mutation_tool_hash = None
        mutated_db = before_db
        execution_failures = 0
        excluded_supervisor_calls = 0
        for call_index, call in enumerate(calls):
            try:
                code, call_fingerprint = _render_rest_call(call)
                output = str(world.execute(code))
            except PermissionError:
                excluded_supervisor_calls += 1
                continue
            except Exception:
                execution_failures += 1
                continue
            if "execution failed" in output.lower() or "traceback" in output.lower():
                execution_failures += 1
                continue
            _, candidate_db = _save_digest(world, f"mutated_{call_index}")
            if candidate_db != before_db:
                mutated_db = candidate_db
                mutation_call_index = call_index
                mutation_tool_hash = call_fingerprint
                break
        if mutation_call_index is None:
            return {
                "task_id_hash": hashlib.sha256(str(task_id).encode()).hexdigest(),
                "passed": False,
                "reason": "no_state_changing_reference_call_observed",
                "reference_calls_examined": len(calls),
                "execution_failures": execution_failures,
                "excluded_supervisor_calls": excluded_supervisor_calls,
                "ground_truth_values_exported": False,
            }

        _compat_load_state(world, before_id)
        loaded_state = True
        _restore_control_state(world, before_control)
        _, restored_db = _save_digest(world, "restored")
        restored_control_hash = _control_digest(_control_state(world))
        db_gate = before_db == restored_db and before_db != mutated_db
        control_gate = before_control_hash == restored_control_hash
        return {
            "task_id_hash": hashlib.sha256(str(task_id).encode()).hexdigest(),
            "passed": bool(db_gate and control_gate),
            "db_gate": db_gate,
            "control_gate": control_gate,
            "before_equals_restored": before_db == restored_db,
            "before_differs_from_mutated": before_db != mutated_db,
            "mutation_call_index": mutation_call_index,
            "mutation_tool_hash": mutation_tool_hash,
            "mutation_scope": "application_api",
            "reference_calls_examined": mutation_call_index + 1,
            "execution_failures": execution_failures,
            "excluded_supervisor_calls": excluded_supervisor_calls,
            "ground_truth_role": "offline_mutation_witness_only",
            "ground_truth_values_exported": False,
            "time_freezer_rearmed_after_load": loaded_state,
        }
    finally:
        world.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--subset", choices=["train", "dev"], default="train")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--minimum-rollbacks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/appworld_rollback_probe")
    )
    args = parser.parse_args()
    os.environ["APPWORLD_ROOT"] = str(args.appworld_root.resolve())

    from appworld import AppWorld, load_task_ids, update_root

    update_root(str(args.appworld_root.resolve()))
    task_ids = list(load_task_ids(args.subset))[: args.limit]
    if not task_ids:
        raise RuntimeError(f"AppWorld subset {args.subset!r} is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records = [
        _probe_task(AppWorld, task_id, args.seed, index)
        for index, task_id in enumerate(task_ids)
    ]
    successful = [record for record in records if record["passed"]]
    passed = len(successful) >= args.minimum_rollbacks
    result = {
        "passed": passed,
        "stage": "appworld_real_rollback_probe",
        "subset": args.subset,
        "num_tasks": len(records),
        "minimum_rollbacks": args.minimum_rollbacks,
        "successful_rollbacks": len(successful),
        "checks": {
            "real_db_mutation_observed": all(
                record.get("before_differs_from_mutated", False)
                for record in successful
            ) and bool(successful),
            "db_restored_exactly": all(
                record.get("before_equals_restored", False) for record in successful
            ) and bool(successful),
            "control_state_restored_exactly": all(
                record.get("control_gate", False) for record in successful
            ) and bool(successful),
            "supervisor_calls_excluded": all(
                record.get("mutation_scope") == "application_api"
                for record in successful
            ),
            "ground_truth_values_not_exported": all(
                not record.get("ground_truth_values_exported", True) for record in records
            ),
        },
        "records": records,
        "wall_time_sec": time.time() - started,
        "authorizes_causal_shadow": passed,
        "next_step": (
            "run the 30-task deployable-harness audit"
            if passed
            else "keep causal Shadow blocked and inspect rollback failures"
        ),
    }
    write_json(args.output_dir / "rollback_probe.json", result)
    write_json(
        args.output_dir / "gate.json",
        {
            key: result[key]
            for key in (
                "passed",
                "stage",
                "successful_rollbacks",
                "checks",
                "authorizes_causal_shadow",
                "next_step",
            )
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # AppWorld installs a custom exception hook that rewrites even SystemExit(0)
    # into an error-looking traceback, so bypass it after flushing artifacts.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if passed else 1)


if __name__ == "__main__":
    main()
