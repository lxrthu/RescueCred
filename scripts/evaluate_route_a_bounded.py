#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import importlib.metadata
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from rescuecredit.appworld_shadow_credit import (
    action_app,
    plain,
    render_compatible_action,
    requirement_progress,
)
from rescuecredit.frozen_bank import (
    digest,
    directory_sha256,
    file_sha256,
    read_jsonl,
    write_jsonl,
)
from rescuecredit.logging import JsonlLogger, write_json
from rescuecredit.route_a_bounded import (
    CONFIRMATORY_CODE_PATHS,
    EXPECTED_EVENTS,
    EXPECTED_EVENT_FILE_SHA256,
    EXPECTED_EVENT_SET_HASH,
    EXPECTED_HORIZONS,
    EXPECTED_SEED,
    continuation_cache_key,
    summarize_bounded_results,
    trace_prefix_matches,
)
from rescuecredit.route_a_immediate import causal_decision
from rescuecredit.route_a_task_eval import event_set_hash
from attach_appworld_shadow_credit import _bounded, _tool_schemas
from audit_appworld_deployable_harness import _render_rest_call


POLICY_LABEL = "azure_gpt4o_temperature0_visible_only_cached_v3_format_repair"
CONTINUATION_REQUEST_KEYS = frozenset(
    {
        "instruction",
        "event_context",
        "tool_schemas",
        "history",
        "remaining_steps",
    }
)
WORKER_ENV_ALLOWLIST = frozenset(
    {
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "DEPLOYMENT_NAME",
        "ENDPOINT_URL",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
    }
)


class WorkerFatalError(RuntimeError):
    pass


class ExclusiveCacheLock:
    def __init__(self, cache_file: Path) -> None:
        self.path = cache_file.with_suffix(cache_file.suffix + ".lock")
        self.handle: Any | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"continuation cache is already locked: {self.path}") from error

    def release(self) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


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


def _policy_identity(worker_script: Path) -> dict[str, Any]:
    azure_client = Path("rescuecredit/azure_client.py")
    endpoint = os.getenv("ENDPOINT_URL", "https://scdall3.openai.azure.com/")
    identity = {
        "policy_label": POLICY_LABEL,
        "worker_script_sha256": file_sha256(worker_script),
        "azure_client_sha256": file_sha256(azure_client),
        "deployment": os.getenv("DEPLOYMENT_NAME", "gpt-4o"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest(),
        "temperature": 0.0,
        "max_tokens": 500,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    identity["fingerprint"] = digest(identity)
    return identity


def _confirmatory_code_identity() -> dict[str, Any]:
    files = {path: file_sha256(Path(path)) for path in CONFIRMATORY_CODE_PATHS}
    return {"files": files, "fingerprint": digest(files)}


def _runtime_identity(
    *, root: Path, AppWorld: Any, events: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    from environments.appworld.adapter import normalize_function_tools

    dependency_paths = [
        Path("scripts/evaluate_route_a_bounded.py"),
        Path("rescuecredit/route_a_bounded.py"),
        Path("scripts/attach_appworld_shadow_credit.py"),
        Path("scripts/audit_appworld_deployable_harness.py"),
        Path("rescuecredit/appworld_shadow_credit.py"),
        Path("environments/appworld/adapter.py"),
    ]
    fixture_records: list[dict[str, Any]] = []
    api_docs_hashes: list[str] = []
    run_tag = f"{os.getpid()}_{time.time_ns()}"
    for event in events:
        task_index = int(event["task_index"])
        world = AppWorld(
            task_id=str(event["task_id"]),
            experiment_name=(
                f"route_a_bounded_identity_{run_tag}_"
                f"{str(event['event_id'])[:12]}_{task_index}"
            ),
            ground_truth_mode="full",
            raise_on_failure=False,
            random_seed=seed + task_index,
        )
        try:
            calls = list(getattr(world.task.ground_truth, "api_calls", []) or [])
            task_api_docs_hash = digest(
                plain(normalize_function_tools(world.task.api_docs))
            )
            api_docs_hashes.append(task_api_docs_hash)
            fixture_records.append(
                {
                    "task_id_hash": hashlib.sha256(
                        str(event["task_id"]).encode()
                    ).hexdigest(),
                    "task_index": task_index,
                    "instruction_hash": hashlib.sha256(
                        str(world.task.instruction).encode()
                    ).hexdigest(),
                    "api_calls_hash": digest(calls),
                    "api_docs_hash": task_api_docs_hash,
                }
            )
        finally:
            world.close()
    dev_split = root / "data" / "datasets" / "dev.txt"
    return {
        "python": sys.version,
        "appworld_version": importlib.metadata.version("appworld"),
        "freezegun_version": importlib.metadata.version("freezegun"),
        "dev_split_sha256": file_sha256(dev_split),
        "task_fixture_hash": digest(fixture_records),
        "public_api_docs_hash": digest(api_docs_hashes),
        "dependency_sha256": {
            str(path): file_sha256(path) for path in dependency_paths
        },
    }


class ContinuationProcess:
    def __init__(
        self,
        python: Path,
        script: Path,
        stderr_path: Path,
        timeout_sec: float,
    ) -> None:
        self.timeout_sec = float(timeout_sec)
        self.stderr = stderr_path.open("w", encoding="utf-8")
        sandbox_dir = Path(tempfile.mkdtemp(prefix="rescuecredit_continuation_"))
        self.sandbox_dir = sandbox_dir
        package_dir = sandbox_dir / "rescuecredit"
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        isolated_script = sandbox_dir / "continuation_worker.py"
        shutil.copy2(script, isolated_script)
        shutil.copy2(
            Path("rescuecredit/appworld_shadow_credit.py"),
            package_dir / "appworld_shadow_credit.py",
        )
        shutil.copy2(
            Path("rescuecredit/azure_client.py"),
            package_dir / "azure_client.py",
        )
        worker_env = {
            key: os.environ[key]
            for key in WORKER_ENV_ALLOWLIST
            if key in os.environ
        }
        worker_env.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            }
        )
        self.process = subprocess.Popen(
            [
                str(python.resolve()),
                str(isolated_script.name),
                "--model",
                "azure-gpt-4o",
                "--device",
                "cpu",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
            cwd=sandbox_dir,
            env=worker_env,
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) != CONTINUATION_REQUEST_KEYS:
            return {"status": "error", "error": "request_allowlist_violation"}
        if self.process.stdin is None or self.process.stdout is None:
            raise WorkerFatalError("worker_pipe_closed")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select(
            [self.process.stdout], [], [], self.timeout_sec
        )
        if not ready:
            self.process.terminate()
            raise WorkerFatalError("worker_timeout")
        raw = self.process.stdout.readline()
        if not raw:
            raise WorkerFatalError("worker_no_response")
        try:
            response = json.loads(raw)
        except json.JSONDecodeError:
            raise WorkerFatalError("worker_invalid_json")
        action = response.get("action")
        if isinstance(action, dict):
            return {"status": "action", "action": action}
        if response.get("stopped") is True:
            return {"status": "stop"}
        return {
            "status": "error",
            "error": str(response.get("error_type") or "worker_invalid_response"),
        }

    def close(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.stderr.close()
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)


class CachedContinuation:
    def __init__(
        self,
        worker: ContinuationProcess,
        cache_file: Path,
        policy_identity: dict[str, Any],
    ) -> None:
        self.worker = worker
        self.cache_file = cache_file
        self.policy_identity = policy_identity
        self.cache: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.conflicts = 0
        if cache_file.is_file():
            for row in read_jsonl(cache_file):
                if row.get("policy_fingerprint") != policy_identity["fingerprint"]:
                    raise ValueError("continuation cache policy fingerprint mismatch")
                key = str(row["key"])
                value = {"status": row["status"], "action": row.get("action")}
                if value["status"] not in {"action", "stop"}:
                    raise ValueError("continuation cache contains a failed response")
                if key in self.cache and self.cache[key] != value:
                    self.conflicts += 1
                self.cache[key] = value
        self.logger = JsonlLogger(cache_file)
        cache_file.touch(exist_ok=True)

    def request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if set(payload) != CONTINUATION_REQUEST_KEYS:
            return {"status": "error", "error": "request_allowlist_violation"}, ""
        key = continuation_cache_key(
            payload, str(self.policy_identity["fingerprint"])
        )
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached, key
        self.misses += 1
        response = self.worker.request(payload)
        # API/protocol failures are never cached and invalidate the branch.
        if response["status"] == "error":
            return response, key
        value = {"status": response["status"], "action": response.get("action")}
        self.cache[key] = value
        self.logger.write(
            {
                "key": key,
                "policy_fingerprint": self.policy_identity["fingerprint"],
                "status": value["status"],
                "action": value.get("action"),
                "protected_reference_values_present": False,
                "evaluator_output_present": False,
            }
        )
        return value, key


def _selection_map(path: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        event_id = str(row["event_id"])
        selected = str(row.get("selected", ""))
        if selected not in {"a", "b"}:
            raise ValueError(f"invalid selection for {event_id}: {selected!r}")
        if event_id in mapping:
            raise ValueError(f"duplicate selection event: {event_id}")
        mapping[event_id] = row
    return mapping


def _validate_protocol(
    *,
    lock_path: Path,
    event_file: Path,
    mask_results: Path,
    v2_results: Path,
    events: list[dict[str, Any]],
    seed: int,
    horizons: list[int],
    confirmatory: bool = False,
    code_identity: dict[str, Any] | None = None,
    policy_identity: dict[str, Any] | None = None,
    runtime_identity: dict[str, Any] | None = None,
    development_protocol: bool = False,
) -> tuple[dict[str, Any], str]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if development_protocol:
        checks = {
            "lock_status": lock.get("status")
            == "frozen_before_both_valid_dev_outcomes",
            "seed": seed == EXPECTED_SEED == lock.get("seed"),
            "horizons": tuple(horizons)
            == EXPECTED_HORIZONS
            == tuple(lock.get("horizons", [])),
            "event_count": len(events) == int(lock.get("events", -1)),
            "unique_event_ids": len({row["event_id"] for row in events})
            == len(events),
            "event_set_hash": event_set_hash(events)
            == lock.get("event_set_hash"),
            "event_file_sha256": file_sha256(event_file)
            == lock.get("event_file_sha256"),
            "mask_results_sha256": file_sha256(mask_results)
            == lock.get("mask_results_sha256"),
            "v2_results_sha256": file_sha256(v2_results)
            == lock.get("v31_results_sha256"),
            "method_binding": lock.get("method_a") == "mask"
            and lock.get("method_b") == "v31",
            "source_identity": bool(lock.get("source_sha256"))
            and all(
                Path(path).is_file()
                and file_sha256(Path(path)) == expected
                for path, expected in lock.get("source_sha256", {}).items()
            ),
            "public_schema_corpus_identity": directory_sha256(
                Path("data/api_docs/openapi")
            )
            == lock.get("public_openapi_schema_corpus_sha256"),
            "all_event_checks": all(lock.get("checks", {}).values()),
        }
    else:
        checks = {
        "lock_status": lock.get("status")
        == (
            "frozen_before_confirmatory_outcomes"
            if confirmatory
            else "frozen_before_bounded_outcomes"
        ),
        "seed": (
            seed in {43, 44, 45} and seed == lock.get("seed")
            if confirmatory
            else seed == EXPECTED_SEED == lock.get("seed")
        ),
        "horizons": tuple(horizons) == EXPECTED_HORIZONS
        == tuple(lock.get("horizons", [])),
        "event_count": len(events) == EXPECTED_EVENTS == lock.get("events"),
        "unique_event_ids": len({row["event_id"] for row in events}) == len(events),
        "event_set_hash": event_set_hash(events)
        == EXPECTED_EVENT_SET_HASH
        == lock.get("event_set_hash"),
        "event_file_sha256": file_sha256(event_file)
        == EXPECTED_EVENT_FILE_SHA256
        == lock.get("event_file_sha256"),
        "mask_results_sha256": file_sha256(mask_results)
        == lock.get("mask_results_sha256"),
        "v2_results_sha256": file_sha256(v2_results)
        == lock.get("v2_results_sha256"),
        "all_event_checks": all(lock.get("checks", {}).values()),
        }
    if confirmatory:
        checks.update(
            {
                "code_identity": code_identity == lock.get("code_identity"),
                "policy_identity": policy_identity == lock.get("policy_identity"),
                "runtime_identity": runtime_identity == lock.get("runtime_identity"),
                "aggregate_gate_frozen": lock.get("gate_frozen_before_outcomes")
                == {
                    "minimum_positive_seeds": 2,
                    "minimum_total_nonzero_events": 15,
                    "require_positive_mean_score_improvement": True,
                    "require_positive_mean_causal_accuracy_improvement": True,
                    "require_aggregate_wins_over_losses": True,
                    "require_cluster_bootstrap_ci_lower_above_zero": True,
                    "bootstrap_clusters": "event_id",
                    "bootstrap_samples": 10000,
                    "bootstrap_seed": 20260717,
                },
            }
        )
    if not all(checks.values()):
        raise ValueError(f"bounded protocol validation failed: {checks}")
    return lock, file_sha256(lock_path)


def _run_branch(
    *,
    AppWorld: Any,
    root: Path,
    event: dict[str, Any],
    action: dict[str, Any],
    branch: str,
    seed: int,
    eval_horizon: int,
    policy_horizon: int,
    run_tag: str,
    continuation: CachedContinuation,
) -> dict[str, Any]:
    task_id = str(event["task_id"])
    task_index = int(event["task_index"])
    experiment_name = (
        f"route_a_bounded_{run_tag}_{str(event['event_id'])[:12]}_"
        f"{branch}_h{eval_horizon}_{seed}_{task_index}"
    )
    world = AppWorld(
        task_id=task_id,
        experiment_name=experiment_name,
        ground_truth_mode="full",
        raise_on_failure=False,
        random_seed=seed + task_index,
    )
    result: dict[str, Any] = {
        "valid": True,
        "score": None,
        "steps": 0,
        "failure_reason": None,
        "action_execution_failed": False,
        "continuation_execution_failures": 0,
        "worker_errors": [],
        "termination": None,
        "trace": [],
    }
    history: list[dict[str, Any]] = []
    try:
        calls = list(getattr(world.task.ground_truth, "api_calls", []) or [])
        call_index = int(event["call_index"])
        if call_index >= len(calls):
            result.update(valid=False, failure_reason="reference_call_index_out_of_range")
            return result
        for prefix_call in calls[:call_index]:
            output = str(world.execute(_render_rest_call(prefix_call)))
            if _execution_failed(output):
                result.update(valid=False, failure_reason="prefix_replay_failed")
                return result
        try:
            output = str(world.execute(render_compatible_action(action)))
            result["action_execution_failed"] = _execution_failed(output)
        except Exception as error:
            result.update(
                valid=False,
                action_execution_failed=True,
                failure_reason=f"initial_action_{type(error).__name__}",
                termination="invalid_initial_action",
            )
            return result
        history.append({"action": action, "output": _bounded(output)})
        result["trace"].append(
            {
                "step": 1,
                "source": "initial",
                "action_hash": digest(action),
                "output_hash": hashlib.sha256(str(output).encode()).hexdigest(),
            }
        )
        active_apps = {action_app(action)}
        steps = 1
        while steps < eval_horizon and not world.task_completed():
            payload = {
                "instruction": str(world.task.instruction),
                "event_context": event.get("continuation_context", event["prompt"]),
                "tool_schemas": _tool_schemas(world, active_apps),
                "history": history[-8:],
                "remaining_steps": policy_horizon - steps,
            }
            response, cache_key = continuation.request(payload)
            if response["status"] == "error":
                result["worker_errors"].append(str(response.get("error")))
                result.update(
                    valid=False,
                    failure_reason=f"continuation_{response.get('error')}",
                    steps=steps,
                    termination="worker_error",
                )
                return result
            if response["status"] == "stop":
                result["trace"].append(
                    {"step": steps + 1, "source": "policy_stop", "cache_key": cache_key}
                )
                result["termination"] = "policy_stop"
                break
            next_action = response.get("action")
            if not isinstance(next_action, dict):
                result.update(
                    valid=False,
                    failure_reason="continuation_missing_action",
                    steps=steps,
                    termination="worker_error",
                )
                return result
            app = action_app(next_action)
            if app:
                active_apps.add(app)
            try:
                output = str(world.execute(render_compatible_action(next_action)))
            except Exception as error:
                result.update(
                    valid=False,
                    failure_reason=f"continuation_action_{type(error).__name__}",
                    steps=steps,
                    termination="invalid_policy_action",
                )
                return result
            if _execution_failed(output):
                result["continuation_execution_failures"] += 1
            history.append({"action": next_action, "output": _bounded(output)})
            steps += 1
            result["trace"].append(
                {
                    "step": steps,
                    "source": "continuation",
                    "cache_key": cache_key,
                    "action_hash": digest(next_action),
                    "output_hash": hashlib.sha256(str(output).encode()).hexdigest(),
                }
            )
        if result["termination"] is None:
            result["termination"] = (
                "task_completed" if world.task_completed() else "horizon"
            )
        save = getattr(world, "save", None)
        if callable(save):
            save()
        world.evaluate()
        score = _report_score(root, experiment_name, task_id)
        if score is None:
            result.update(valid=False, failure_reason="official_report_score_missing")
            return result
        result.update(score=float(score), steps=steps)
        return result
    except WorkerFatalError:
        raise
    except Exception as error:
        result.update(valid=False, failure_reason=type(error).__name__)
        return result
    finally:
        try:
            world.close()
        except Exception as error:
            result["close_error_type"] = type(error).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("--mask-results", type=Path, required=True)
    parser.add_argument("--v2-results", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(EXPECTED_HORIZONS))
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--worker-timeout-sec", type=float, default=120.0)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=[3])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--confirmatory",
        action="store_true",
        help="Authorize only preregistered follow-up seeds 43/44/45 with a confirmatory lock.",
    )
    parser.add_argument(
        "--development-protocol",
        action="store_true",
        help="Validate a dynamically frozen both-valid dev event set before outcomes.",
    )
    args = parser.parse_args()

    if args.confirmatory and args.development_protocol:
        raise ValueError("confirmatory and development protocols are mutually exclusive")

    horizons = sorted(set(args.horizons))
    allowed_seed = (
        args.seed in {43, 44, 45}
        if args.confirmatory
        else args.seed == EXPECTED_SEED
    )
    if not allowed_seed or tuple(horizons) != EXPECTED_HORIZONS:
        raise ValueError(
            "this preregistered diagnostic requires seed=42 and horizons=4 8; "
            "--confirmatory permits only frozen seeds 43/44/45 with the same horizons"
        )
    root = args.appworld_root.resolve()
    os.environ["APPWORLD_ROOT"] = str(root)
    from appworld import AppWorld, update_root

    update_root(str(root))
    events = read_jsonl(args.event_file)
    code_identity = _confirmatory_code_identity()
    policy_identity = _policy_identity(args.worker_script)
    runtime_identity = _runtime_identity(
        root=root, AppWorld=AppWorld, events=events, seed=args.seed
    )
    lock, lock_sha256 = _validate_protocol(
        lock_path=args.protocol_lock,
        event_file=args.event_file,
        mask_results=args.mask_results,
        v2_results=args.v2_results,
        events=events,
        seed=args.seed,
        horizons=horizons,
        confirmatory=args.confirmatory,
        code_identity=code_identity,
        policy_identity=policy_identity,
        runtime_identity=runtime_identity,
        development_protocol=args.development_protocol,
    )
    mask = _selection_map(args.mask_results)
    v2 = _selection_map(args.v2_results)
    if set(mask) != {row["event_id"] for row in events} or set(v2) != {
        row["event_id"] for row in events
    }:
        raise ValueError("selection maps do not exactly match the frozen event set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_lock = ExclusiveCacheLock(args.cache_file)
    cache_lock.acquire()
    atexit.register(cache_lock.release)
    worker = ContinuationProcess(
        args.worker_python,
        args.worker_script,
        args.output_dir / "continuation_stderr.log",
        args.worker_timeout_sec,
    )
    continuation = CachedContinuation(worker, args.cache_file, policy_identity)
    rows: list[dict[str, Any]] = []
    started = time.time()
    run_tag = f"{os.getpid()}_{time.time_ns()}"
    implementation_fingerprint = digest(
        {
            "evaluator_sha256": file_sha256(Path(__file__)),
            "summary_module_sha256": file_sha256(
                Path("rescuecredit/route_a_bounded.py")
            ),
            "protocol_lock_sha256": lock_sha256,
            "policy_fingerprint": policy_identity["fingerprint"],
            "runtime_identity": runtime_identity,
            "code_identity": code_identity,
        }
    )
    checkpoint_dir = args.output_dir / "event_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    evaluation_events = events[: args.limit] if args.limit else events
    resumed_events = 0
    try:
        for index, event in enumerate(evaluation_events):
            event_id = str(event["event_id"])
            checkpoint_path = checkpoint_dir / f"{event_id}.json"
            if checkpoint_path.is_file():
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if (
                    checkpoint.get("event_id") != event_id
                    or checkpoint.get("row", {}).get("event_id") != event_id
                    or checkpoint_path.stem != event_id
                    or checkpoint.get("implementation_fingerprint")
                    != implementation_fingerprint
                    or checkpoint.get("protocol_lock_sha256") != lock_sha256
                ):
                    raise ValueError("event checkpoint fingerprint mismatch")
                rows.append(checkpoint["row"])
                resumed_events += 1
                continue
            branch_runs: dict[str, dict[str, dict[str, Any]]] = {}
            horizon_payload: dict[str, Any] = {}
            for horizon in horizons:
                branch_runs[str(horizon)] = {}
                for branch, action in (("a", event["action_a"]), ("b", event["action_b"])):
                    branch_runs[str(horizon)][branch] = _run_branch(
                        AppWorld=AppWorld,
                        root=root,
                        event=event,
                        action=action,
                        branch=branch,
                        seed=args.seed,
                        eval_horizon=horizon,
                        policy_horizon=max(horizons),
                        run_tag=run_tag,
                        continuation=continuation,
                    )
                a = branch_runs[str(horizon)]["a"]
                b = branch_runs[str(horizon)]["b"]
                valid = bool(a["valid"] and b["valid"])
                score_a = a.get("score")
                score_b = b.get("score")
                horizon_payload[str(horizon)] = {
                    "evaluation_valid": valid,
                    "score_a": score_a,
                    "score_b": score_b,
                    "delta": float(score_b) - float(score_a) if valid else None,
                    "decision": (
                        causal_decision(float(score_a), float(score_b)) if valid else None
                    ),
                    "steps_a": a["steps"],
                    "steps_b": b["steps"],
                    "termination_a": a["termination"],
                    "termination_b": b["termination"],
                    "action_a_execution_failed": a["action_execution_failed"],
                    "action_b_execution_failed": b["action_execution_failed"],
                    "continuation_execution_failures": int(
                        a["continuation_execution_failures"]
                    )
                    + int(b["continuation_execution_failures"]),
                    "branch_a_failure_reason": a["failure_reason"],
                    "branch_b_failure_reason": b["failure_reason"],
                }
            prefix_a = trace_prefix_matches(
                branch_runs["4"]["a"], branch_runs["8"]["a"]
            )
            prefix_b = trace_prefix_matches(
                branch_runs["4"]["b"], branch_runs["8"]["b"]
            )
            row = {
                "event_id": event_id,
                "task_id_hash": hashlib.sha256(str(event["task_id"]).encode()).hexdigest(),
                "task_index": event["task_index"],
                "mask_selected": mask[event_id]["selected"],
                "v2_selected": v2[event_id]["selected"],
                "mask_margin": mask[event_id].get("b_over_a_margin"),
                "v2_margin": v2[event_id].get("b_over_a_margin"),
                "action_a_hash": digest(event["action_a"]),
                "action_b_hash": digest(event["action_b"]),
                "horizons": horizon_payload,
                "horizon_prefix_match_a": prefix_a,
                "horizon_prefix_match_b": prefix_b,
                "trace_hashes": {
                    horizon: {
                        branch: digest(branch_runs[horizon][branch]["trace"])
                        for branch in ("a", "b")
                    }
                    for horizon in ("4", "8")
                },
                "protocol_lock_sha256": lock_sha256,
                "protected_reference_values_exported": False,
                "reference_suffix_used": False,
            }
            rows.append(row)
            write_json(
                checkpoint_path,
                {
                    "event_id": event_id,
                    "implementation_fingerprint": implementation_fingerprint,
                    "protocol_lock_sha256": lock_sha256,
                    "row": row,
                },
            )
            print(
                json.dumps(
                    {
                        "progress": f"{index + 1}/{len(evaluation_events)}",
                        "primary_valid": sum(
                            item["horizons"]["8"]["evaluation_valid"] for item in rows
                        ),
                        "primary_nonzero": sum(
                            item["horizons"]["8"]["evaluation_valid"]
                            and abs(float(item["horizons"]["8"]["delta"])) > 1e-12
                            for item in rows
                        ),
                        "prefix_mismatches": sum(
                            not item["horizon_prefix_match_a"]
                            or not item["horizon_prefix_match_b"]
                            for item in rows
                        ),
                        "cache_hits": continuation.hits,
                        "cache_misses": continuation.misses,
                    }
                ),
                flush=True,
            )
    finally:
        worker.close()

    result_path = args.output_dir / "bounded_results.jsonl"
    write_jsonl(result_path, rows)
    summary = summarize_bounded_results(
        rows, horizons=horizons, event_set_hash=event_set_hash(events)
    )
    summary.update(
        {
            "seed": args.seed,
            "stage": f"route_a_appworld_bounded_horizon_seed{args.seed}",
            "confirmatory": args.confirmatory,
            "development_protocol": args.development_protocol,
            "requested_horizons": horizons,
            "run_tag": run_tag,
            "sanity_limit": args.limit,
            "implementation_fingerprint": implementation_fingerprint,
            "resumed_events": resumed_events,
            "protocol_lock_validated": True,
            "protocol_lock_sha256": lock_sha256,
            "protocol_lock": lock,
            "code_identity": code_identity,
            "policy_identity": policy_identity,
            "runtime_identity": runtime_identity,
            "continuation_request_allowlist": sorted(CONTINUATION_REQUEST_KEYS),
            "worker_environment_allowlist": sorted(WORKER_ENV_ALLOWLIST),
            "worker_cwd_isolated": True,
            "worker_sandbox_location": "system_tmp_outside_appworld_root",
            "worker_benchmark_root_in_environment": False,
            "event_file_sha256": file_sha256(args.event_file),
            "mask_results_sha256": file_sha256(args.mask_results),
            "v2_results_sha256": file_sha256(args.v2_results),
            "bounded_results_sha256": file_sha256(result_path),
            "bounded_results_rows": len(rows),
            "bounded_results_event_set_hash": event_set_hash(rows),
            "cache_file": str(args.cache_file),
            "cache_sha256": file_sha256(args.cache_file),
            "cache_entries": len(continuation.cache),
            "cache_hits": continuation.hits,
            "cache_misses": continuation.misses,
            "cache_conflicts": continuation.conflicts,
            "worker_failures_are_invalid": True,
            "official_report_required": True,
            "wall_time_sec": time.time() - started,
        }
    )
    write_json(args.output_dir / "bounded_summary.json", summary)
    cache_lock.release()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
