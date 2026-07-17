#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.route_a_task_eval import event_set_hash


class AdapterScorer:
    def __init__(
        self,
        *,
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
            raise RuntimeError("scorer pipes are unavailable")
        payload = {
            "prompt": event["prompt"],
            "action_a": event["action_a"],
            "action_b": event["action_b"],
        }
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        raw = self.process.stdout.readline()
        if not raw:
            raise RuntimeError("scorer exited without a response")
        return json.loads(raw)

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.stderr.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mask", "v3"], required=True)
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--scorer-script", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    events = read_jsonl(args.event_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = AdapterScorer(
        python=args.worker_python,
        script=args.scorer_script,
        model=args.model,
        adapter=args.adapter,
        stderr_path=args.output_dir / "scorer_stderr.log",
    )
    rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for index, event in enumerate(events):
            response = scorer.score(event)
            selected = str(response.get("selected", ""))
            scoring_failed = bool(response.get("scoring_failed"))
            if selected not in {"a", "b"}:
                selected = "a"
                scoring_failed = True
            rows.append(
                {
                    "event_id": str(event["event_id"]),
                    "method": args.method,
                    "selected": selected,
                    "b_over_a_margin": response.get("b_over_a_margin"),
                    "scoring_failed": scoring_failed,
                }
            )
            if (index + 1) % 10 == 0 or index + 1 == len(events):
                print(
                    json.dumps(
                        {
                            "progress": f"{index + 1}/{len(events)}",
                            "selected_b": sum(row["selected"] == "b" for row in rows),
                            "failures": sum(row["scoring_failed"] for row in rows),
                        }
                    ),
                    flush=True,
                )
    finally:
        scorer.close()

    result_path = args.output_dir / "task_results.jsonl"
    write_jsonl(result_path, rows)
    summary = {
        "status": "completed",
        "stage": "route_a_frozen_dev_selection",
        "method": args.method,
        "events": len(rows),
        "event_set_hash": event_set_hash(events),
        "event_file_sha256": file_sha256(args.event_file),
        "results_sha256": file_sha256(result_path),
        "selected_b": sum(row["selected"] == "b" for row in rows),
        "scoring_failures": sum(row["scoring_failed"] for row in rows),
        "model": str(args.model),
        "adapter": str(args.adapter),
        "reference_free_model_inputs": True,
        "test_split_access": False,
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "selection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
