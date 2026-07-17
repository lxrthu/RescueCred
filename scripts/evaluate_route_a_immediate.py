#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from rescuecredit.appworld_shadow_credit import (
    official_score,
    prefix_replay_failed,
    render_compatible_action,
    requirement_progress,
)
from rescuecredit.frozen_bank import digest, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_immediate import (
    causal_decision,
    summarize_immediate_results,
)
from rescuecredit.route_a_task_eval import event_set_hash
from audit_appworld_deployable_harness import _render_rest_call


def _execution_failed(output: str) -> bool:
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in ("execution failed", "traceback", "response status code is")
    )


def _report_score(root: Path, experiment_name: str, task_id: str) -> float | None:
    report = (
        root
        / "experiments"
        / "outputs"
        / experiment_name
        / "tasks"
        / task_id
        / "evaluation"
        / "report.md"
    )
    if not report.is_file():
        return None
    try:
        return requirement_progress(
            report.read_text(encoding="utf-8", errors="replace")
        )[2]
    except ValueError:
        return None


def _run_branch(
    *,
    AppWorld: Any,
    root: Path,
    event: dict[str, Any],
    action: dict[str, Any],
    branch: str,
    seed: int,
    index: int,
    run_tag: str,
) -> dict[str, Any]:
    task_id = str(event["task_id"])
    experiment_name = f"route_a_immediate_{run_tag}_{branch}_{seed}_{index}"
    world = AppWorld(
        task_id=task_id,
        experiment_name=experiment_name,
        ground_truth_mode="full",
        raise_on_failure=False,
        random_seed=seed + index,
    )
    result: dict[str, Any] = {
        "valid": True,
        "score": None,
        "failure_reason": None,
        "action_execution_failed": False,
    }
    try:
        calls = list(getattr(world.task.ground_truth, "api_calls", []) or [])
        call_index = int(event["call_index"])
        if call_index >= len(calls):
            result.update(valid=False, failure_reason="reference_call_index_out_of_range")
            return result
        for prefix_call in calls[:call_index]:
            output = str(world.execute(_render_rest_call(prefix_call)))
            if prefix_replay_failed(output):
                result.update(valid=False, failure_reason="prefix_replay_failed")
                return result
        try:
            output = str(world.execute(render_compatible_action(action)))
            result["action_execution_failed"] = _execution_failed(output)
        except Exception as error:
            result["action_execution_failed"] = True
            result["action_error_type"] = type(error).__name__
        save = getattr(world, "save", None)
        if callable(save):
            save()
        evaluation = world.evaluate()
        score = _report_score(root, experiment_name, task_id)
        if score is None:
            score = official_score(evaluation)
        if score is None:
            result.update(valid=False, failure_reason="official_score_missing")
            return result
        result["score"] = float(score)
        return result
    except Exception as error:
        result.update(
            valid=False,
            failure_reason=f"{type(error).__name__}",
        )
        return result
    finally:
        try:
            world.close()
        except Exception as error:
            result["close_error_type"] = type(error).__name__


def _selection_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        event_id = str(row["event_id"])
        selected = str(row.get("selected", ""))
        if selected not in {"a", "b"}:
            raise ValueError(f"invalid selection for {event_id}: {selected!r}")
        if event_id in mapping:
            raise ValueError(f"duplicate selection event: {event_id}")
        mapping[event_id] = row
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("--mask-results", type=Path, required=True)
    parser.add_argument("--v2-results", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.appworld_root.resolve()
    os.environ["APPWORLD_ROOT"] = str(root)
    from appworld import AppWorld, update_root

    update_root(str(root))
    events = read_jsonl(args.event_file)
    mask = _selection_map(args.mask_results)
    v2 = _selection_map(args.v2_results)
    missing_mask = [row["event_id"] for row in events if row["event_id"] not in mask]
    missing_v2 = [row["event_id"] for row in events if row["event_id"] not in v2]
    if missing_mask or missing_v2:
        raise ValueError(
            f"selection files do not cover events: mask={len(missing_mask)} v2={len(missing_v2)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.time()
    run_tag = f"{os.getpid()}_{time.time_ns()}"
    for index, event in enumerate(events):
        event_id = str(event["event_id"])
        branch_a = _run_branch(
            AppWorld=AppWorld,
            root=root,
            event=event,
            action=event["action_a"],
            branch="a",
            seed=args.seed,
            index=index,
            run_tag=run_tag,
        )
        branch_b = _run_branch(
            AppWorld=AppWorld,
            root=root,
            event=event,
            action=event["action_b"],
            branch="b",
            seed=args.seed,
            index=index,
            run_tag=run_tag,
        )
        valid = bool(branch_a["valid"] and branch_b["valid"])
        score_a = branch_a.get("score")
        score_b = branch_b.get("score")
        decision = (
            causal_decision(float(score_a), float(score_b)) if valid else None
        )
        row = {
            "event_id": event_id,
            "task_id_hash": hashlib.sha256(str(event["task_id"]).encode()).hexdigest(),
            "evaluation_valid": valid,
            "score_a": score_a,
            "score_b": score_b,
            "delta": float(score_b) - float(score_a) if valid else None,
            "decision": decision,
            "mask_selected": mask[event_id]["selected"],
            "v2_selected": v2[event_id]["selected"],
            "mask_margin": mask[event_id].get("b_over_a_margin"),
            "v2_margin": v2[event_id].get("b_over_a_margin"),
            "action_a_hash": digest(event["action_a"]),
            "action_b_hash": digest(event["action_b"]),
            "action_a_execution_failed": branch_a["action_execution_failed"],
            "action_b_execution_failed": branch_b["action_execution_failed"],
            "branch_a_failure_reason": branch_a["failure_reason"],
            "branch_b_failure_reason": branch_b["failure_reason"],
            "continuation_used": False,
            "reference_suffix_used": False,
            "azure_used": False,
            "protected_reference_values_exported": False,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "progress": f"{index + 1}/{len(events)}",
                    "valid": sum(item["evaluation_valid"] for item in rows),
                    "nonzero": sum(
                        item["evaluation_valid"]
                        and abs(float(item["delta"])) > 1e-12
                        for item in rows
                    ),
                    "selection_disagreements": sum(
                        item["mask_selected"] != item["v2_selected"]
                        for item in rows
                    ),
                }
            ),
            flush=True,
        )

    output_rows = args.output_dir / "immediate_results.jsonl"
    write_jsonl(output_rows, rows)
    summary = summarize_immediate_results(
        rows,
        event_set_hash=event_set_hash(events),
    )
    summary.update(
        {
            "seed": args.seed,
            "run_tag": run_tag,
            "event_file": str(args.event_file),
            "mask_results": str(args.mask_results),
            "v2_results": str(args.v2_results),
            "event_file_sha256": hashlib.sha256(
                args.event_file.read_bytes()
            ).hexdigest(),
            "wall_time_sec": time.time() - started,
        }
    )
    write_json(args.output_dir / "immediate_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
