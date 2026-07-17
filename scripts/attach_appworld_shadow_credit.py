#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from environments.appworld.adapter import normalize_function_tools
from rescuecredit.appworld_shadow_credit import (
    action_app,
    credit_decision,
    official_score,
    prefix_replay_failed,
    render_compatible_action,
)
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from audit_appworld_deployable_harness import _render_rest_call


def _failed(output: str) -> bool:
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in ("execution failed", "traceback", "response status code is")
    )


def _bounded(text: Any, limit: int = 3000) -> str:
    value = str(text)
    return value if len(value) <= limit else value[:limit] + "...[truncated]"


class ContinuationWorker:
    def __init__(self, python: Path, script: Path, stderr_path: Path) -> None:
        self.stderr = stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(python), str(script), "--model", "azure-gpt-4o", "--device", "cpu"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )

    def next(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if self.process.stdin is None or self.process.stdout is None:
            return None, "worker_pipe_closed"
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        raw = self.process.stdout.readline()
        if not raw:
            return None, "worker_no_response"
        response = json.loads(raw)
        action = response.get("action")
        return (action if isinstance(action, dict) else None), response.get("error_type")

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.stderr.close()


def _tool_schemas(world: Any, active_apps: set[str]) -> list[dict[str, Any]]:
    tools = normalize_function_tools(world.task.api_docs)
    helper_apps = {"api_docs", "supervisor"}
    selected = []
    for tool in tools:
        app = str(tool["name"]).split("__", 1)[0]
        if app not in active_apps | helper_apps:
            continue
        selected.append(
            {
                "name": tool["name"],
                "required": tool["required"],
                "optional": tool["optional"],
                "parameter_schema": tool["parameter_schema"],
                "description": _bounded(tool["description"], 500),
            }
        )
    return selected


def _run_branch(
    *,
    AppWorld: Any,
    record: dict[str, Any],
    branch: str,
    action: dict[str, Any],
    reference_prefix: list[dict[str, Any]],
    worker: ContinuationWorker,
    seed: int,
    max_steps: int,
    experiment_name: str,
) -> dict[str, Any]:
    world = AppWorld(
        task_id=record["task_id"],
        experiment_name=experiment_name,
        ground_truth_mode="minimal",
        raise_on_failure=False,
        random_seed=seed,
    )
    history: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        # Controlled-state fixture only. These actions never enter a model prompt,
        # bank record, credit record, or training file.
        for prefix_action in reference_prefix:
            output = str(world.execute(_render_rest_call(prefix_action)))
            if prefix_replay_failed(output):
                return {"replay_valid": False, "reason": "prefix_execution_failed", "steps": 0}

        output = str(world.execute(render_compatible_action(action)))
        history.append({"action": action, "output": _bounded(output)})
        active_apps = {action_app(action)}
        steps = 1
        while steps < max_steps and not world.task_completed():
            next_action, error = worker.next(
                {
                    "instruction": str(world.task.instruction),
                    "event_context": record["prompt"],
                    "tool_schemas": _tool_schemas(world, active_apps),
                    "history": history[-8:],
                    "remaining_steps": max_steps - steps,
                    "branch": branch,
                }
            )
            if error:
                errors.append(error)
            if next_action is None:
                break
            app = action_app(next_action)
            if app:
                active_apps.add(app)
            try:
                output = str(world.execute(render_compatible_action(next_action)))
            except Exception as exc:
                errors.append(type(exc).__name__)
                break
            history.append({"action": next_action, "output": _bounded(output)})
            steps += 1
        save = getattr(world, "save", None)
        if callable(save):
            save()
        try:
            score = official_score(world.evaluate())
        except Exception as exc:
            return {
                "replay_valid": False,
                "reason": f"official_evaluator_{type(exc).__name__}",
                "steps": steps,
            }
        if score is None:
            return {"replay_valid": False, "reason": "official_score_missing", "steps": steps}
        return {
            "replay_valid": True,
            "return": score,
            "steps": steps,
            "worker_errors": errors,
            "claimed_complete": bool(world.task_completed()),
        }
    finally:
        world.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach reference-free frozen-policy Shadow credit to Route-A events"
    )
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-shadow-steps", type=int, default=6)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.appworld_root.resolve()
    os.environ["APPWORLD_ROOT"] = str(root)
    from appworld import AppWorld, update_root

    update_root(str(root))
    bank_path = args.bank_dir / "correction_bank.public.jsonl"
    records = read_jsonl(bank_path)[args.offset : args.offset + args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker = ContinuationWorker(
        args.worker_python,
        args.worker_script,
        args.output_dir / "continuation_stderr.log",
    )
    credits: list[dict[str, Any]] = []
    failures = Counter()
    started = time.time()
    try:
        for index, record in enumerate(records):
            fixture = AppWorld(
                task_id=record["task_id"],
                experiment_name=f"route_a_fixture_{args.seed}_{index}",
                ground_truth_mode="full",
                raise_on_failure=False,
                random_seed=args.seed + index,
            )
            try:
                calls = list(getattr(fixture.task.ground_truth, "api_calls", []) or [])
                reference_prefix = calls[: int(record["call_index"])]
            finally:
                fixture.close()
            a = _run_branch(
                AppWorld=AppWorld,
                record=record,
                branch="A",
                action=record["action_a"],
                reference_prefix=reference_prefix,
                worker=worker,
                seed=args.seed * 1000 + index,
                max_steps=args.max_shadow_steps,
                experiment_name=f"route_a_shadow_a_{args.seed}_{index}",
            )
            b = _run_branch(
                AppWorld=AppWorld,
                record=record,
                branch="B",
                action=record["action_b"],
                reference_prefix=reference_prefix,
                worker=worker,
                seed=args.seed * 1000 + index,
                max_steps=args.max_shadow_steps,
                experiment_name=f"route_a_shadow_b_{args.seed}_{index}",
            )
            if not a.get("replay_valid") or not b.get("replay_valid"):
                reason = str(a.get("reason") or b.get("reason") or "unknown")
                failures[reason] += 1
                continue
            return_a = float(a["return"])
            return_b = float(b["return"])
            credits.append(
                {
                    "event_id": record["event_id"],
                    "return_a": return_a,
                    "return_b": return_b,
                    "delta": return_b - return_a,
                    "decision": credit_decision(return_a, return_b),
                    "steps_a": int(a["steps"]),
                    "steps_b": int(b["steps"]),
                    "replay_valid": True,
                    "continuation_policy": "azure_gpt4o_temperature0_visible_only_v1",
                }
            )
            print(
                json.dumps(
                    {
                        "progress": f"{index + 1}/{len(records)}",
                        "valid": len(credits),
                        "nonzero": sum(abs(row["delta"]) > 1e-9 for row in credits),
                    }
                ),
                flush=True,
            )
    finally:
        worker.close()

    credit_path = args.output_dir / "shadow_credit.train.jsonl"
    write_jsonl(credit_path, credits)
    decisions = Counter(row["decision"] for row in credits)
    summary = {
        "status": "completed",
        "bank_sha256": file_sha256(bank_path),
        "requested_events": len(records),
        "valid_events": len(credits),
        "nonzero_events": sum(abs(row["delta"]) > 1e-9 for row in credits),
        "decisions": dict(decisions),
        "failure_reasons": dict(failures),
        "total_shadow_steps": sum(row["steps_a"] + row["steps_b"] for row in credits),
        "state_fixture": "train reference prefix, never exposed to continuation policy or training",
        "offline_audit_private_read": False,
        "continuation_inputs": [
            "task_instruction",
            "frozen_public_event_context",
            "public_api_schemas",
            "visible_branch_history",
        ],
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "shadow_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
