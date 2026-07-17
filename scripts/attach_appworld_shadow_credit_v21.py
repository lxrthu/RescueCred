#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rescuecredit.appworld_shadow_credit import credit_decision
from rescuecredit.frozen_bank import digest, file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import JsonlLogger, write_json
try:
    from attach_appworld_shadow_credit import _run_branch
except ModuleNotFoundError:
    from scripts.attach_appworld_shadow_credit import _run_branch


class WorkerFatalError(RuntimeError):
    """Fatal failure of the isolated continuation worker.

    This exception belongs to the V2.1 isolation wrapper.  The shared legacy
    branch helper intentionally does not define it, so keep the boundary local
    instead of requiring a particular version of the legacy module.
    """


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


class IsolatedCachedWorker:
    def __init__(
        self,
        python: Path,
        script: Path,
        stderr_path: Path,
        cache_path: Path,
        timeout_sec: float,
    ) -> None:
        self.timeout_sec = float(timeout_sec)
        self.sandbox = Path(tempfile.mkdtemp(prefix="rescuecredit_v21_worker_"))
        package = self.sandbox / "rescuecredit"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        isolated_script = self.sandbox / "continuation_worker.py"
        shutil.copy2(script, isolated_script)
        shutil.copy2(
            Path("rescuecredit/appworld_shadow_credit.py"),
            package / "appworld_shadow_credit.py",
        )
        shutil.copy2(
            Path("rescuecredit/azure_client.py"), package / "azure_client.py"
        )
        worker_env = {
            key: os.environ[key]
            for key in WORKER_ENV_ALLOWLIST
            if key in os.environ
        }
        worker_env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        self.stderr = stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(python.resolve()),
                isolated_script.name,
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
            cwd=self.sandbox,
            env=worker_env,
        )
        self.policy_fingerprint = digest(
            {
                "worker_sha256": file_sha256(script),
                "azure_client_sha256": file_sha256(
                    Path("rescuecredit/azure_client.py")
                ),
                "temperature": 0.0,
                "max_tokens": 500,
                "request_contract": "visible_only_v21",
            }
        )
        self.cache: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.conflicts = 0
        if cache_path.is_file():
            for row in read_jsonl(cache_path):
                if row.get("policy_fingerprint") != self.policy_fingerprint:
                    raise ValueError("V2.1 cache policy fingerprint mismatch")
                key = str(row["key"])
                value = {"status": row["status"], "action": row.get("action")}
                if key in self.cache and self.cache[key] != value:
                    self.conflicts += 1
                self.cache[key] = value
        cache_path.touch(exist_ok=True)
        self.logger = JsonlLogger(cache_path)

    def next(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        visible = dict(payload)
        visible.pop("branch", None)
        key = digest(
            {"policy_fingerprint": self.policy_fingerprint, "request": visible}
        )
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached.get("action"), None
        self.misses += 1
        if self.process.stdin is None or self.process.stdout is None:
            return None, "worker_pipe_closed"
        self.process.stdin.write(json.dumps(visible, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout_sec)
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
            value = {"status": "action", "action": action}
        elif response.get("stopped") is True:
            value = {"status": "stop", "action": None}
        else:
            return None, str(response.get("error_type") or "worker_invalid_response")
        self.cache[key] = value
        self.logger.write(
            {
                "key": key,
                "policy_fingerprint": self.policy_fingerprint,
                "status": value["status"],
                "action": value.get("action"),
                "reference_or_evaluator_input_present": False,
            }
        )
        return value.get("action"), None

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
        shutil.rmtree(self.sandbox, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach isolated reference-free Shadow credit to the V2.1 bank"
    )
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--max-shadow-steps", type=int, default=12)
    parser.add_argument("--worker-timeout-sec", type=float, default=120.0)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.appworld_root.resolve()
    os.environ["APPWORLD_ROOT"] = str(root)
    from appworld import AppWorld, update_root

    update_root(str(root))
    bank_path = args.bank_dir / "correction_bank.public.jsonl"
    all_records = read_jsonl(bank_path)
    stop = None if args.limit is None else args.offset + args.limit
    records = all_records[args.offset : stop]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "event_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    worker = IsolatedCachedWorker(
        args.worker_python,
        args.worker_script,
        args.output_dir / "continuation_stderr.log",
        args.output_dir / "continuation_cache.jsonl",
        args.worker_timeout_sec,
    )
    fingerprint = digest(
        {
            "script_sha256": file_sha256(Path(__file__)),
            "branch_helper_sha256": file_sha256(
                Path("scripts/attach_appworld_shadow_credit.py")
            ),
            "bank_sha256": file_sha256(bank_path),
            "policy_fingerprint": worker.policy_fingerprint,
            "seed": args.seed,
            "max_shadow_steps": args.max_shadow_steps,
        }
    )
    credits: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    started = time.time()
    try:
        for local_index, record in enumerate(records):
            global_index = args.offset + local_index
            checkpoint_path = checkpoint_dir / f"{record['event_id']}.json"
            if checkpoint_path.is_file():
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if checkpoint.get("fingerprint") != fingerprint:
                    raise ValueError("V2.1 checkpoint fingerprint mismatch")
                if checkpoint["status"] == "valid":
                    credits.append(checkpoint["credit"])
                else:
                    failures[str(checkpoint["reason"])] += 1
                continue
            fixture = AppWorld(
                task_id=record["task_id"],
                experiment_name=f"route_a_v21_fixture_{args.seed}_{global_index}",
                ground_truth_mode="full",
                raise_on_failure=False,
                random_seed=args.seed + global_index,
            )
            try:
                calls = list(getattr(fixture.task.ground_truth, "api_calls", []) or [])
                reference_prefix = calls[: int(record["call_index"])]
            finally:
                fixture.close()
            branch_seed = args.seed * 1000 + global_index
            a = _run_branch(
                AppWorld=AppWorld,
                record=record,
                branch="A",
                action=record["action_a"],
                reference_prefix=reference_prefix,
                worker=worker,
                seed=branch_seed,
                max_steps=args.max_shadow_steps,
                experiment_name=f"route_a_shadow_a_{args.seed}_{global_index}",
            )
            b = _run_branch(
                AppWorld=AppWorld,
                record=record,
                branch="B",
                action=record["action_b"],
                reference_prefix=reference_prefix,
                worker=worker,
                seed=branch_seed,
                max_steps=args.max_shadow_steps,
                experiment_name=f"route_a_shadow_b_{args.seed}_{global_index}",
            )
            if not a.get("replay_valid") or not b.get("replay_valid"):
                reason = str(a.get("reason") or b.get("reason") or "unknown")
                failures[reason] += 1
                write_json(
                    checkpoint_path,
                    {"fingerprint": fingerprint, "status": "invalid", "reason": reason},
                )
                continue
            return_a = float(a["return"])
            return_b = float(b["return"])
            credit = {
                "event_id": record["event_id"],
                "return_a": return_a,
                "return_b": return_b,
                "delta": return_b - return_a,
                "decision": credit_decision(return_a, return_b),
                "steps_a": int(a["steps"]),
                "steps_b": int(b["steps"]),
                "replay_valid": True,
                "continuation_policy": "azure_gpt4o_temperature0_visible_only_v21",
            }
            credits.append(credit)
            write_json(
                checkpoint_path,
                {"fingerprint": fingerprint, "status": "valid", "credit": credit},
            )
            print(
                json.dumps(
                    {
                        "progress": f"{local_index + 1}/{len(records)}",
                        "valid": len(credits),
                        "nonzero_binary": sum(
                            abs(row["delta"]) > 1e-12 for row in credits
                        ),
                        "cache_hits": worker.hits,
                        "cache_misses": worker.misses,
                    }
                ),
                flush=True,
            )
    finally:
        worker.close()

    credit_path = args.output_dir / "shadow_credit.train.jsonl"
    write_jsonl(credit_path, sorted(credits, key=lambda row: row["event_id"]))
    decisions = Counter(row["decision"] for row in credits)
    summary = {
        "status": "completed",
        "bank_sha256": file_sha256(bank_path),
        "requested_events": len(records),
        "valid_events": len(credits),
        "replay_valid_rate": len(credits) / max(1, len(records)),
        "nonzero_events": sum(abs(row["delta"]) > 1e-12 for row in credits),
        "decisions": dict(decisions),
        "failure_reasons": dict(failures),
        "total_shadow_steps": sum(row["steps_a"] + row["steps_b"] for row in credits),
        "cache_hits": worker.hits,
        "cache_misses": worker.misses,
        "cache_conflicts": worker.conflicts,
        "worker_cwd_isolated": True,
        "worker_benchmark_root_in_environment": False,
        "worker_environment_allowlist": sorted(WORKER_ENV_ALLOWLIST),
        "state_fixture": "train reference prefix; never exported to policy or training",
        "offline_audit_private_read": False,
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "shadow_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
