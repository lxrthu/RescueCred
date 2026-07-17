#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rescuecredit.appworld_shadow_credit import (
    action_app,
    official_score,
    prefix_replay_failed,
    render_compatible_action,
    requirement_progress,
)
from rescuecredit.frozen_bank import digest, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_task_eval import event_set_hash, summarize_task_results
from attach_appworld_shadow_credit import ContinuationWorker, _tool_schemas
from audit_appworld_deployable_harness import _canonical, _render_rest_call


def _execution_failed(output: str) -> bool:
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in ("execution failed", "traceback", "response status code is")
    )


def _bounded(value: Any, limit: int = 3000) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


class AdapterScorerWorker:
    def __init__(
        self,
        python: Path,
        script: Path,
        model: Path,
        adapter: Path,
        stderr_path: Path,
    ) -> None:
        self.stderr = stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(python),
                str(script),
                "--model",
                str(model),
                "--adapter",
                str(adapter),
                "--fp32",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )

    def score(self, event: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            return {"action": None, "selected": "a", "scoring_failed": True}
        payload = {
            "prompt": event["prompt"],
            "action_a": event["action_a"],
            "action_b": event["action_b"],
        }
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            return {"action": None, "selected": "a", "scoring_failed": True}
        return json.loads(response)

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.stderr.close()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--method", choices=["mask", "v2"], required=True)
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--scorer-script", type=Path, required=True)
    parser.add_argument("--continuation-script", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--max-continuation-steps", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.appworld_root.resolve()
    os.environ["APPWORLD_ROOT"] = str(root)
    from appworld import AppWorld, update_root

    update_root(str(root))
    events = read_jsonl(args.event_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = AdapterScorerWorker(
        args.worker_python,
        args.scorer_script,
        args.model,
        args.adapter,
        args.output_dir / "scorer_stderr.log",
    )
    continuation = ContinuationWorker(
        args.worker_python,
        args.continuation_script,
        args.output_dir / "continuation_stderr.log",
    )
    results: list[dict[str, Any]] = []
    started = time.time()
    try:
        for index, event in enumerate(events):
            response = scorer.score(event)
            scoring_failed = bool(response.get("scoring_failed"))
            selected = str(response.get("selected", "a"))
            action = response.get("action")
            if not isinstance(action, dict):
                selected = "a"
                action = event["action_a"]
                scoring_failed = True
            task_id = str(event["task_id"])
            experiment_name = f"route_a_dev_v2_{args.method}_{args.seed}_{index}"
            world = AppWorld(
                task_id=task_id,
                experiment_name=experiment_name,
                ground_truth_mode="full",
                raise_on_failure=False,
                random_seed=args.seed + index,
            )
            evaluation_valid = True
            failure_reason = None
            candidate_execution_failed = False
            continuation_errors: list[str] = []
            continuation_steps = 0
            score = None
            correction_matches_reference = False
            try:
                calls = list(getattr(world.task.ground_truth, "api_calls", []) or [])
                call_index = int(event["call_index"])
                if call_index >= len(calls):
                    evaluation_valid = False
                    failure_reason = "reference_call_index_out_of_range"
                else:
                    expected = _canonical(calls[call_index])
                    correction_matches_reference = action == expected
                    for prefix_call in calls[:call_index]:
                        output = str(world.execute(_render_rest_call(prefix_call)))
                        if prefix_replay_failed(output):
                            evaluation_valid = False
                            failure_reason = "prefix_replay_failed"
                            break
                    history: list[dict[str, Any]] = []
                    if evaluation_valid:
                        try:
                            output = str(world.execute(render_compatible_action(action)))
                            candidate_execution_failed = _execution_failed(output)
                        except Exception as error:
                            output = f"{type(error).__name__}: {error}"
                            candidate_execution_failed = True
                        history.append({"action": action, "output": _bounded(output)})
                        active_apps = {action_app(action)}
                        total_steps = 1
                        while (
                            total_steps < args.max_continuation_steps
                            and not world.task_completed()
                        ):
                            next_action, error = continuation.next(
                                {
                                    "instruction": str(world.task.instruction),
                                    "event_context": event["prompt"],
                                    "tool_schemas": _tool_schemas(world, active_apps),
                                    "history": history[-8:],
                                    "remaining_steps": args.max_continuation_steps
                                    - total_steps,
                                    "branch": args.method,
                                }
                            )
                            if error:
                                continuation_errors.append(str(error))
                            if next_action is None:
                                break
                            app = action_app(next_action)
                            if app:
                                active_apps.add(app)
                            try:
                                output = str(
                                    world.execute(render_compatible_action(next_action))
                                )
                            except Exception as error:
                                continuation_errors.append(type(error).__name__)
                                break
                            history.append(
                                {"action": next_action, "output": _bounded(output)}
                            )
                            total_steps += 1
                            continuation_steps += 1
                    if evaluation_valid:
                        save = getattr(world, "save", None)
                        if callable(save):
                            save()
                        evaluation = world.evaluate()
                        score = _report_score(root, experiment_name, task_id)
                        if score is None:
                            score = official_score(evaluation)
                        if score is None:
                            evaluation_valid = False
                            failure_reason = "official_score_missing"
            finally:
                world.close()
            results.append(
                {
                    "event_id": event["event_id"],
                    "task_id_hash": hashlib.sha256(task_id.encode()).hexdigest(),
                    "method": args.method,
                    "evaluation_valid": evaluation_valid,
                    "official_score": score,
                    "task_success": bool(score is not None and score >= 1.0 - 1e-12),
                    "adapter_scoring_failed": scoring_failed,
                    "selected": selected,
                    "b_over_a_margin": response.get("b_over_a_margin"),
                    "candidate_execution_failed": candidate_execution_failed,
                    "correction_matches_reference": correction_matches_reference,
                    "continuation_steps": continuation_steps,
                    "continuation_errors": continuation_errors,
                    "candidate_hash": digest(action),
                    "failure_reason": failure_reason,
                    "protected_reference_values_sent_to_workers": False,
                    "reference_suffix_used": False,
                }
            )
            print(
                json.dumps(
                    {
                        "progress": f"{index + 1}/{len(events)}",
                        "valid": sum(row["evaluation_valid"] for row in results),
                        "success": sum(row["task_success"] for row in results),
                        "selected_b": sum(row["selected"] == "b" for row in results),
                    }
                ),
                flush=True,
            )
    finally:
        scorer.close()
        continuation.close()

    results_path = args.output_dir / "task_results.jsonl"
    write_jsonl(results_path, results)
    summary = summarize_task_results(
        results, method=args.method, events_hash=event_set_hash(events)
    )
    summary.pop("generation_failure_rate", None)
    summary.update(
        {
            "seed": args.seed,
            "adapter": str(args.adapter),
            "adapter_scoring_failure_rate": sum(
                row["adapter_scoring_failed"] for row in results
            )
            / max(1, len(results)),
            "selected_b_rate": sum(row["selected"] == "b" for row in results)
            / max(1, len(results)),
            "mean_continuation_steps": sum(
                row["continuation_steps"] for row in results
            )
            / max(1, len(results)),
            "continuation_error_counts": dict(
                Counter(
                    error for row in results for error in row["continuation_errors"]
                )
            ),
            "test_split_access": False,
            "worker_input_contract": (
                "adapter sees public prompt plus A/B; continuation sees visible state/history only"
            ),
            "reference_suffix_used": False,
            "wall_time_sec": time.time() - started,
        }
    )
    write_json(args.output_dir / "eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
