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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from environments.toolsandbox import (
    TOOL_SANDBOX_COMMIT,
    V4_SCENARIO_POOL_PROFILE,
    ToolSandboxRuntime,
    controlled_missing_argument,
    score_decision,
)
from rescuecredit.toolsandbox_audit import DEFAULT_THRESHOLDS, build_summary_and_gate
from rescuecredit.toolsandbox_credit import (
    LEXICOGRAPHIC_COMPONENT_ORDER,
    lexicographic_counterfactual_regret,
    validate_branch_credit_evidence,
)
from rescuecredit.toolsandbox_protocol import (
    REQUIRED_V4_SOURCE_PATHS,
    current_toolsandbox_runtime_identity,
)


WORKER_ENV_ALLOWLIST = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_THINKING",
    "DEPLOYMENT_NAME",
    "ENDPOINT_URL",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "TOOLSANDBOX_LLM_PROVIDER",
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


def _validate_protocol_lock(
    path: Path,
    args: argparse.Namespace,
    worker_script: Path,
) -> tuple[str, Dict[str, Any]]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "frozen_before_v4_outcomes",
        "toolsandbox_commit": TOOL_SANDBOX_COMMIT,
        "seed": args.seed,
        "scenario_offset": args.scenario_offset,
        "limit": args.limit,
        "horizon": args.horizon,
        "event_search_steps": args.event_search_steps,
        "credit_mode": args.credit_mode,
        "scenario_pool_profile": V4_SCENARIO_POOL_PROFILE,
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"protocol lock mismatch for {key}: {lock.get(key)!r} != {value!r}")
    if lock.get("provider") != os.getenv("TOOLSANDBOX_LLM_PROVIDER", "azure"):
        raise ValueError("protocol provider does not match worker environment")
    expected_provider = os.getenv("TOOLSANDBOX_LLM_PROVIDER", "azure")
    if expected_provider == "deepseek":
        expected_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        expected_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        expected_thinking = os.getenv("DEEPSEEK_THINKING", "disabled")
        if lock.get("model") != expected_model:
            raise ValueError("protocol model does not match worker environment")
        if lock.get("base_url") != expected_base_url:
            raise ValueError("protocol base URL does not match worker environment")
        if lock.get("thinking") != expected_thinking:
            raise ValueError("protocol thinking mode does not match worker environment")
    if lock.get("lexicographic_component_order") != list(
        LEXICOGRAPHIC_COMPONENT_ORDER
    ):
        raise ValueError("protocol component order does not match implementation")
    if float(lock.get("atol", -1.0)) != 1e-12:
        raise ValueError("protocol tolerance mismatch")
    if lock.get("thresholds") != dict(DEFAULT_THRESHOLDS):
        raise ValueError("protocol thresholds do not match implementation")
    plan = Path(str(lock.get("plan", "")))
    if not plan.is_file() or _sha256(plan) != lock.get("plan_sha256"):
        raise ValueError("protocol plan identity mismatch")
    source_sha256 = lock.get("source_sha256")
    if not isinstance(source_sha256, Mapping):
        raise ValueError("protocol source hashes are missing")
    if set(source_sha256) != set(REQUIRED_V4_SOURCE_PATHS):
        raise ValueError("protocol source hash inventory is incomplete or unexpected")
    root = Path(__file__).resolve().parents[1]
    for relative, expected_hash in source_sha256.items():
        source = root / str(relative)
        if not source.is_file() or _sha256(source) != expected_hash:
            raise ValueError(f"protocol source identity mismatch: {relative}")
    if _sha256(worker_script) != source_sha256.get("scripts/toolsandbox_azure_worker.py"):
        raise ValueError("worker identity does not match protocol")
    stage0 = Path(str(lock.get("stage0_gate", "")))
    if not stage0.is_file() or _sha256(stage0) != lock.get("stage0_gate_sha256"):
        raise ValueError("Stage-0 gate identity mismatch")
    stage0_payload = json.loads(stage0.read_text(encoding="utf-8"))
    if stage0_payload.get("passed") is not True:
        raise ValueError("Stage-0 gate is not passed")
    if stage0_payload.get("official_commit") != TOOL_SANDBOX_COMMIT:
        raise ValueError("Stage-0 ToolSandbox commit mismatch")
    if lock.get("toolsandbox_runtime") != current_toolsandbox_runtime_identity(
        TOOL_SANDBOX_COMMIT
    ):
        raise ValueError("ToolSandbox runtime identity mismatch")
    scenario_identity = lock.get("scenario_identity")
    if not isinstance(scenario_identity, Mapping):
        raise ValueError("protocol scenario identity is missing")
    if scenario_identity.get("intersection") != []:
        raise ValueError("protocol development/fresh scenario sets overlap")
    return _sha256(path), dict(lock)


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
    visible_receipt: Optional[Mapping[str, Any]] = None,
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


def _official_score_readonly(
    runtime: ToolSandboxRuntime, scenario: Any, context: Any
) -> Dict[str, Any]:
    before = runtime.context_digest(context)
    score = runtime.official_score(scenario, context)
    after = runtime.context_digest(context)
    if before != after:
        raise RuntimeError("official evaluator mutated branch continuation state")
    return score


def _snapshot_audit_exact(
    snapshot_checks: int,
    snapshot_mismatches: int,
    branch_prefix_checks: int,
    branch_prefix_mismatches: int,
) -> bool:
    return (
        snapshot_checks > 0
        and snapshot_mismatches == 0
        and branch_prefix_checks > 0
        and branch_prefix_mismatches == 0
    )


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
    score_trace: List[Dict[str, Any]] = []
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
        score_trace.append(_official_score_readonly(runtime, scenario, context))
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
            score_trace.append(_official_score_readonly(runtime, scenario, context))
        score = score_trace[-1]
        similarity_trace = [float(item["similarity"]) for item in score_trace]
        padded_similarity_trace = similarity_trace + [similarity_trace[-1]] * (
            horizon - len(similarity_trace)
        )
        return {
            "valid": True,
            "score": score,
            "steps": len(receipts),
            "tool_errors": sum(bool(receipt.get("exception")) for receipt in receipts),
            "score_trace": score_trace,
            "padded_similarity_trace": padded_similarity_trace,
            "progress_auc": sum(padded_similarity_trace) / float(horizon),
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
    credit_mode: str,
    horizon: int,
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
    if credit_mode == "terminal":
        credit = {
            "decision": score_decision(delta) if delta is not None else "invalid",
            "decision_basis": (
                "final_official_similarity" if delta is not None else "invalid_replay"
            ),
            "decision_value": delta,
            "causal_weight": min(1.0, abs(delta)) if delta is not None else 0.0,
            "components": (
                {"final_official_similarity": delta} if delta is not None else {}
            ),
        }
    elif credit_mode == "lexicographic_v4":
        if valid:
            validate_branch_credit_evidence(branch_a, horizon=horizon)
            validate_branch_credit_evidence(branch_b, horizon=horizon)
        credit = lexicographic_counterfactual_regret(
            branch_a, branch_b, horizon=horizon
        )
    else:
        raise ValueError(f"unsupported credit mode: {credit_mode}")
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
        "terminal_delta": delta,
        "decision": credit["decision"],
        "decision_basis": credit["decision_basis"],
        "decision_value": credit["decision_value"],
        "causal_weight": credit["causal_weight"],
        "credit_components": credit["components"],
        "credit_mode": credit_mode,
        "branch_a": branch_a,
        "branch_b": branch_b,
        **dict(metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--scenario-offset", type=int, default=0)
    parser.add_argument(
        "--credit-mode",
        choices=("terminal", "lexicographic_v4"),
        default="terminal",
    )
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--worker-model")
    parser.add_argument("--worker-device")
    parser.add_argument("--worker-timeout-sec", type=float, default=180.0)
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument(
        "--event-search-steps",
        type=int,
        default=8,
        help=(
            "Maximum reference-free common-prefix actions to inspect for the first "
            "controlled or natural treatment point."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    worker_script = args.worker_script.resolve()
    worker_python = args.worker_python.resolve()
    if not worker_script.is_file() or not worker_python.is_file():
        raise FileNotFoundError("worker python/script is missing")
    protocol_lock_sha256 = None
    protocol_lock_payload: Optional[Dict[str, Any]] = None
    if args.protocol_lock is not None:
        if args.credit_mode != "lexicographic_v4":
            raise ValueError("V4 protocol lock requires lexicographic_v4 credit")
        protocol_lock_sha256, protocol_lock_payload = _validate_protocol_lock(
            args.protocol_lock.resolve(), args, worker_script
        )

    runtime = ToolSandboxRuntime()
    selected = runtime.select_scenarios(
        limit=args.limit,
        seed=args.seed,
        offset=args.scenario_offset,
        allow_distraction_tools=(args.credit_mode == "lexicographic_v4"),
    )
    selected_scenario_hashes = [
        hashlib.sha256(name.encode("utf-8")).hexdigest() for name, _ in selected
    ]
    protocol_selected_identity_validated = False
    if protocol_lock_payload is not None:
        scenario_identity = protocol_lock_payload["scenario_identity"]
        if selected_scenario_hashes != scenario_identity.get("fresh_hashes"):
            raise ValueError("selected fresh scenario identity does not match protocol")
        if set(selected_scenario_hashes) & set(
            scenario_identity.get("development_hashes", [])
        ):
            raise ValueError("selected fresh scenarios overlap protocol development set")
        protocol_selected_identity_validated = True
    rows: List[Dict[str, Any]] = []
    worker_failures = 0
    snapshot_checks = 0
    snapshot_mismatches = 0
    branch_prefix_checks = 0
    branch_prefix_mismatches = 0
    mutation_observed = False
    proposal_stats: Counter[str] = Counter()
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
            task_hash = hashlib.sha256(scenario_name.encode("utf-8")).hexdigest()
            treatment_found = False
            search_steps = min(args.horizon, max(1, args.event_search_steps))
            examined = 0
            for prefix_step in range(search_steps):
                examined += 1
                schemas = runtime.tool_schemas(prefix)
                proposal_stats["proposal_requests"] += 1
                try:
                    proposal_response = worker.request(
                        _worker_payload(runtime, prefix, "propose", args.horizon)
                    )
                except Exception as error:
                    worker_failures += 1
                    proposal_stats["proposal_worker_failures"] += 1
                    print(
                        json.dumps(
                            {
                                "progress": f"{index}/{len(selected)}",
                                "proposal": "worker_failure",
                                "prefix_step": prefix_step,
                                "error": type(error).__name__,
                            }
                        ),
                        flush=True,
                    )
                    break
                proposal = _action_or_none(proposal_response)
                if proposal is None:
                    stopped = proposal_response.get("stopped", False) is True
                    proposal_stats[
                        "proposal_stops" if stopped else "proposal_invalid"
                    ] += 1
                    worker_failures += int(not stopped)
                    break
                proposal_stats["proposal_actions"] += 1

                # Execute once on an isolated snapshot. This both verifies exact
                # restore semantics and supplies the visible receipt for the
                # natural Harness track. No reference continuation is involved.
                before = runtime.context_digest(prefix)
                restored = runtime.snapshot(prefix)
                snapshot_checks += 1
                if before != runtime.context_digest(restored):
                    snapshot_mismatches += 1
                natural_probe = runtime.execute(runtime.snapshot(prefix), proposal)
                mutation_observed = mutation_observed or (
                    before != runtime.context_digest(natural_probe.context)
                )

                controlled = controlled_missing_argument(proposal, schemas)
                if controlled is not None:
                    proposal_stats["controlled_eligible"] += 1
                    action_a, removed = controlled
                    branch_a = _run_branch(
                        runtime, scenario, prefix, action_a, worker, args.horizon
                    )
                    branch_prefix_checks += 1
                    if runtime.context_digest(prefix) != before:
                        branch_prefix_mismatches += 1
                    branch_b = _run_branch(
                        runtime, scenario, prefix, proposal, worker, args.horizon
                    )
                    branch_prefix_checks += 1
                    if runtime.context_digest(prefix) != before:
                        branch_prefix_mismatches += 1
                    rows.append(
                        _paired_row(
                            "controlled_missing_argument",
                            scenario_name,
                            task_hash,
                            action_a,
                            proposal,
                            branch_a,
                            branch_b,
                            {
                                "removed_public_required_field": removed,
                                "reference_free_prefix_steps": prefix_step,
                            },
                            args.credit_mode,
                            args.horizon,
                        )
                    )
                    treatment_found = True
                else:
                    proposal_stats["controlled_ineligible"] += 1

                # Natural Harness audit: request B only after A emits a visible
                # execution error. The prefix and receipt are policy-visible.
                if natural_probe.exception:
                    proposal_stats["visible_execution_errors"] += 1
                    try:
                        repair_response = worker.request(
                            _worker_payload(
                                runtime,
                                prefix,
                                "repair",
                                args.horizon,
                                proposal_a=proposal,
                                visible_receipt={
                                    "content": natural_probe.content,
                                    "tool_call_exception": natural_probe.exception,
                                },
                            )
                        )
                    except Exception:
                        worker_failures += 1
                        proposal_stats["repair_worker_failures"] += 1
                        break
                    repair = _action_or_none(repair_response)
                    if repair is not None and repair != proposal:
                        proposal_stats["natural_repairs"] += 1
                        branch_a = _run_branch(
                            runtime, scenario, prefix, proposal, worker, args.horizon
                        )
                        branch_prefix_checks += 1
                        if runtime.context_digest(prefix) != before:
                            branch_prefix_mismatches += 1
                        branch_b = _run_branch(
                            runtime, scenario, prefix, repair, worker, args.horizon
                        )
                        branch_prefix_checks += 1
                        if runtime.context_digest(prefix) != before:
                            branch_prefix_mismatches += 1
                        rows.append(
                            _paired_row(
                                "natural_visible_error_repair",
                                scenario_name,
                                task_hash,
                                proposal,
                                repair,
                                branch_a,
                                branch_b,
                                {
                                    "visible_error": natural_probe.content,
                                    "reference_free_prefix_steps": prefix_step,
                                },
                                args.credit_mode,
                                args.horizon,
                            )
                        )
                        treatment_found = True
                    elif repair_response.get("error_type"):
                        worker_failures += 1
                        proposal_stats["repair_invalid"] += 1
                    else:
                        proposal_stats["repair_abstentions"] += 1

                if treatment_found:
                    break
                prefix = natural_probe.context
                proposal_stats["reference_free_prefix_advances"] += 1
            if not treatment_found:
                proposal_stats["scenarios_without_treatment"] += 1

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
                        "prefix_actions_examined": examined,
                        "treatment_found": treatment_found,
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
    official_evaluator_used = bool(valid_rows)
    for row in valid_rows:
        for branch in (row.get("branch_a", {}), row.get("branch_b", {})):
            if branch.get("score", {}).get("source") != (
                "official ToolSandbox EvaluationResult.similarity"
            ):
                official_evaluator_used = False
                continue
            if args.credit_mode == "lexicographic_v4":
                try:
                    validate_branch_credit_evidence(branch, horizon=args.horizon)
                except (TypeError, ValueError):
                    official_evaluator_used = False

    snapshot_restore_exact = _snapshot_audit_exact(
        snapshot_checks,
        snapshot_mismatches,
        branch_prefix_checks,
        branch_prefix_mismatches,
    )
    protocol_required = args.credit_mode == "lexicographic_v4"
    protocol_validated = (
        protocol_lock_sha256 is not None and protocol_selected_identity_validated
    )
    summary, gate = build_summary_and_gate(
        rows,
        scenarios_requested=args.limit,
        scenarios_selected=len(selected),
        worker_failures=worker_failures,
        snapshot_restore_exact=snapshot_restore_exact,
        official_evaluator_used=official_evaluator_used,
        protocol_required=protocol_required,
        protocol_validated=protocol_validated,
    )
    summary.update(
        {
            "seed": args.seed,
            "scenario_offset": args.scenario_offset,
            "horizon": args.horizon,
            "credit_mode": args.credit_mode,
            "protocol_lock_sha256": protocol_lock_sha256,
            "protocol_validated": protocol_validated,
            "toolsandbox_commit": TOOL_SANDBOX_COMMIT,
            "worker_python": str(worker_python),
            "worker_script_sha256": _sha256(worker_script),
            "event_search": {
                "maximum_reference_free_prefix_steps": min(
                    args.horizon, max(1, args.event_search_steps)
                ),
                "statistics": dict(sorted(proposal_stats.items())),
                "reference_actions_used": False,
            },
            "snapshot_audit": {
                "snapshot_checks": snapshot_checks,
                "snapshot_mismatches": snapshot_mismatches,
                "branch_prefix_checks": branch_prefix_checks,
                "branch_prefix_mismatches": branch_prefix_mismatches,
                "mutation_observed": mutation_observed,
                "exact": snapshot_restore_exact,
            },
            "worker_llm": {
                "provider": os.getenv("TOOLSANDBOX_LLM_PROVIDER", "azure"),
                "model": (
                    os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
                    if os.getenv("TOOLSANDBOX_LLM_PROVIDER", "azure") == "deepseek"
                    else os.getenv("DEPLOYMENT_NAME", "gpt-4o")
                ),
                "base_url": (
                    os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
                    if os.getenv("TOOLSANDBOX_LLM_PROVIDER", "azure") == "deepseek"
                    else os.getenv("ENDPOINT_URL", "https://scdall3.openai.azure.com/")
                ),
                "thinking": (
                    os.getenv("DEEPSEEK_THINKING", "disabled")
                    if os.getenv("TOOLSANDBOX_LLM_PROVIDER", "azure") == "deepseek"
                    else None
                ),
                "secret_exported": False,
            },
            "event_file_sha256": _sha256(events_path),
            "selected_scenario_hashes": selected_scenario_hashes,
            "wall_time_sec": time.time() - started,
        }
    )
    gate.update(
        {
            "credit_mode": args.credit_mode,
            "protocol_lock_sha256": protocol_lock_sha256,
            "protocol_validated": protocol_validated,
            "fresh_scenario_identity_validated": (
                protocol_selected_identity_validated if protocol_required else None
            ),
            "snapshot_audit": summary["snapshot_audit"],
        }
    )
    _write_json(output / "audit_summary.json", summary)
    _write_json(output / "quality_gate.json", gate)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
