#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from environments.toolsandbox import (
    TOOL_SANDBOX_COMMIT,
    ToolSandboxRuntime,
    controlled_missing_argument,
    score_decision,
)
from rescuecredit.toolsandbox_audit import build_summary_and_gate


WORKER_ENV_ALLOWLIST = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "DEPLOYMENT_NAME",
    "ENDPOINT_URL",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class Worker:
    def __init__(
        self,
        python: Path,
        script: Path,
        stderr_path: Path,
        model: Optional[str] = None,
        device: Optional[str] = None,
        timeout_sec: float = 180.0,
    ) -> None:
        self.timeout_sec = float(timeout_sec)
        self._stderr = stderr_path.open("w", encoding="utf-8")
        environment = {key: os.environ[key] for key in WORKER_ENV_ALLOWLIST if key in os.environ}
        environment["PYTHONUNBUFFERED"] = "1"
        command = [str(python), str(script)]
        if model:
            command += ["--model", model]
        if device:
            command += ["--device", device]
        self._sandbox = tempfile.TemporaryDirectory(prefix="rescuecredit-toolsandbox-worker-")
        self.process = subprocess.Popen(
            command,
            cwd=self._sandbox.name,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._reader = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def request(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("worker pipes are unavailable")
        if self.process.poll() is not None:
            raise RuntimeError("worker exited before request")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        future = self._reader.submit(self.process.stdout.readline)
        try:
            raw = future.result(timeout=self.timeout_sec)
        except concurrent.futures.TimeoutError as error:
            self.process.terminate()
            self.process.wait(timeout=5)
            raise TimeoutError(
                "worker response exceeded " + str(self.timeout_sec) + " seconds"
            ) from error
        if not raw:
            raise RuntimeError("worker returned EOF")
        response = json.loads(raw)
        if not isinstance(response, dict):
            raise TypeError("worker response must be an object")
        return response

    def close(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        self._reader.shutdown(wait=True, cancel_futures=True)
        self._stderr.close()
        self._sandbox.cleanup()


def _worker_payload(
    runtime: ToolSandboxRuntime,
    context: Any,
    mode: str,
    remaining_steps: int,
    proposal_a: Optional[Mapping[str, Any]] = None,
    visible_receipt: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "mode": mode,
        "history": runtime.visible_history(context),
        "tool_schemas": runtime.tool_schemas(context),
        "remaining_steps": remaining_steps,
        "proposal_a": proposal_a,
        "visible_receipt": visible_receipt,
    }


def _action_or_none(response: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    action = response.get("action")
    return dict(action) if isinstance(action, Mapping) else None


def _run_branch(
    runtime: ToolSandboxRuntime,
    scenario: Any,
    prefix: Any,
    first_action: Mapping[str, Any],
    worker: Worker,
    horizon: int,
) -> Dict[str, Any]:
    context = runtime.snapshot(prefix)
    receipts: List[Dict[str, Any]] = []
    try:
        receipt = runtime.execute(context, first_action)
        context = receipt.context
        receipts.append(
            {
                "action": receipt.action,
                "content": receipt.content,
                "exception": receipt.exception,
            }
        )
        for step in range(1, horizon):
            try:
                response = worker.request(
                    _worker_payload(
                        runtime,
                        context,
                        mode="continue",
                        remaining_steps=horizon - step,
                    )
                )
            except Exception as error:
                return {
                    "valid": False,
                    "failure_reason": "continuation_worker_failure",
                    "worker_error": type(error).__name__,
                    "receipts": receipts,
                }
            if response.get("stopped") is True:
                break
            action = _action_or_none(response)
            if action is None:
                return {
                    "valid": False,
                    "failure_reason": "continuation_worker_failure",
                    "worker_error": response.get("error_type"),
                    "receipts": receipts,
                }
            receipt = runtime.execute(context, action)
            context = receipt.context
            receipts.append(
                {
                    "action": receipt.action,
                    "content": receipt.content,
                    "exception": receipt.exception,
                }
            )
        score = runtime.official_score(scenario, context)
        return {
            "valid": True,
            "score": score,
            "steps": len(receipts),
            "receipts": receipts,
            "ending_context_digest": runtime.context_digest(context),
        }
    except Exception as error:
        return {
            "valid": False,
            "failure_reason": type(error).__name__,
            "receipts": receipts,
        }


def _paired_row(
    mode: str,
    scenario_name: str,
    task_hash: str,
    action_a: Mapping[str, Any],
    action_b: Mapping[str, Any],
    branch_a: Mapping[str, Any],
    branch_b: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    valid = branch_a.get("valid") is True and branch_b.get("valid") is True
    score_a = (
        float(branch_a["score"]["similarity"])
        if valid and isinstance(branch_a.get("score"), Mapping)
        else None
    )
    score_b = (
        float(branch_b["score"]["similarity"])
        if valid and isinstance(branch_b.get("score"), Mapping)
        else None
    )
    delta = score_b - score_a if score_a is not None and score_b is not None else None
    return {
        "event_id": hashlib.sha256(
            (mode + "\0" + scenario_name).encode("utf-8")
        ).hexdigest(),
        "scenario_name": scenario_name,
        "task_id_hash": task_hash,
        "mode": mode,
        "action_a": action_a,
        "action_b": action_b,
        "replay_valid": valid,
        "return_a": score_a,
        "return_b": score_b,
        "delta": delta,
        "decision": score_decision(delta) if delta is not None else "invalid",
        "branch_a": branch_a,
        "branch_b": branch_b,
        **dict(metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--worker-model")
    parser.add_argument("--worker-device")
    parser.add_argument("--worker-timeout-sec", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    worker_script = args.worker_script.resolve()
    worker_python = args.worker_python.resolve()
    if not worker_script.is_file() or not worker_python.is_file():
        raise FileNotFoundError("worker python/script is missing")

    runtime = ToolSandboxRuntime()
    selected = runtime.select_scenarios(limit=args.limit, seed=args.seed)
    rows: List[Dict[str, Any]] = []
    worker_failures = 0
    snapshot_restore_exact = False
    worker = Worker(
        worker_python,
        worker_script,
        output / "worker_stderr.log",
        args.worker_model,
        args.worker_device,
        args.worker_timeout_sec,
    )
    try:
        for index, (scenario_name, scenario) in enumerate(selected, start=1):
            prefix = runtime.prepare(scenario)
            schemas = runtime.tool_schemas(prefix)
            try:
                proposal_response = worker.request(
                    _worker_payload(runtime, prefix, "propose", args.horizon)
                )
            except Exception as error:
                worker_failures += 1
                print(
                    json.dumps(
                        {
                            "progress": f"{index}/{len(selected)}",
                            "proposal": "worker_failure",
                            "error": type(error).__name__,
                        }
                    ),
                    flush=True,
                )
                continue
            proposal = _action_or_none(proposal_response)
            if proposal is None:
                worker_failures += int(not proposal_response.get("stopped", False))
                print(
                    json.dumps(
                        {"progress": f"{index}/{len(selected)}", "proposal": "none"}
                    ),
                    flush=True,
                )
                continue

            if not snapshot_restore_exact:
                before = runtime.context_digest(prefix)
                restored = runtime.snapshot(prefix)
                mutation = runtime.execute(runtime.snapshot(prefix), proposal).context
                snapshot_restore_exact = (
                    before == runtime.context_digest(restored)
                    and before != runtime.context_digest(mutation)
                )

            task_hash = hashlib.sha256(scenario_name.encode("utf-8")).hexdigest()
            controlled = controlled_missing_argument(proposal, schemas)
            if controlled is not None:
                action_a, removed = controlled
                branch_a = _run_branch(
                    runtime, scenario, prefix, action_a, worker, args.horizon
                )
                branch_b = _run_branch(
                    runtime, scenario, prefix, proposal, worker, args.horizon
                )
                rows.append(
                    _paired_row(
                        "controlled_missing_argument",
                        scenario_name,
                        task_hash,
                        action_a,
                        proposal,
                        branch_a,
                        branch_b,
                        {"removed_public_required_field": removed},
                    )
                )

            # Natural Harness audit: correction B is requested only when proposal A
            # actually emits a visible execution error in an isolated branch.
            natural_probe = runtime.execute(runtime.snapshot(prefix), proposal)
            if natural_probe.exception:
                try:
                    repair_response = worker.request(
                        _worker_payload(
                            runtime,
                            prefix,
                            "repair",
                            args.horizon,
                            proposal_a=proposal,
                            visible_receipt=natural_probe.content,
                        )
                    )
                except Exception:
                    worker_failures += 1
                    continue
                repair = _action_or_none(repair_response)
                if repair is not None and repair != proposal:
                    branch_a = _run_branch(
                        runtime, scenario, prefix, proposal, worker, args.horizon
                    )
                    branch_b = _run_branch(
                        runtime, scenario, prefix, repair, worker, args.horizon
                    )
                    rows.append(
                        _paired_row(
                            "natural_visible_error_repair",
                            scenario_name,
                            task_hash,
                            proposal,
                            repair,
                            branch_a,
                            branch_b,
                            {"visible_error": natural_probe.content},
                        )
                    )
                elif repair_response.get("error_type"):
                    worker_failures += 1

            nonzero = sum(
                row.get("replay_valid") and row.get("decision") != "zero_delta"
                for row in rows
            )
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(selected)}",
                        "events": len(rows),
                        "nonzero": nonzero,
                    }
                ),
                flush=True,
            )
    finally:
        worker.close()

    events_path = output / "signal_events.jsonl"
    events_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    worker_failures += sum(
        branch.get("failure_reason") == "continuation_worker_failure"
        for row in rows
        for branch in (row.get("branch_a", {}), row.get("branch_b", {}))
        if isinstance(branch, Mapping)
    )
    valid_rows = [row for row in rows if row.get("replay_valid") is True]
    official_evaluator_used = bool(valid_rows) and all(
        branch.get("score", {}).get("source")
        == "official ToolSandbox EvaluationResult.similarity"
        for row in valid_rows
        for branch in (row.get("branch_a", {}), row.get("branch_b", {}))
    )
    summary, gate = build_summary_and_gate(
        rows,
        scenarios_requested=args.limit,
        scenarios_selected=len(selected),
        worker_failures=worker_failures,
        snapshot_restore_exact=snapshot_restore_exact,
        official_evaluator_used=official_evaluator_used,
    )
    summary.update(
        {
            "seed": args.seed,
            "horizon": args.horizon,
            "toolsandbox_commit": TOOL_SANDBOX_COMMIT,
            "worker_python": str(worker_python),
            "worker_script_sha256": _sha256(worker_script),
            "event_file_sha256": _sha256(events_path),
            "selected_scenario_hashes": [
                hashlib.sha256(name.encode("utf-8")).hexdigest() for name, _ in selected
            ],
            "wall_time_sec": time.time() - started,
        }
    )
    _write_json(output / "audit_summary.json", summary)
    _write_json(output / "quality_gate.json", gate)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
