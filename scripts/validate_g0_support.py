#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import pvariance

from environments.api_bank import APIBankControlledEnv
from rescuecredit.logging import write_json


def run_recovery(task: dict, off_path: dict, recover: bool) -> float:
    env = APIBankControlledEnv(task["max_steps"])
    env.reset(task, seed=20260714)
    _, reward, done, _ = env.step(off_path)
    if done:
        return reward
    if recover:
        for action in task["reference_actions"]:
            _, reward, done, _ = env.step(action)
            if done:
                return reward
    return env.step({"type": "finish"})[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, default=Path("data/api_bank_controlled_v1/dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sanity/g0_support.json"))
    args = parser.parse_args()
    tasks = [json.loads(line) for line in args.data_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    returns: list[float] = []
    eligible = 0
    for task in tasks:
        expected_tool = task["reference_actions"][0]["tool"]
        off_path = next((action for action in task["reference_actions"] if action["tool"] != expected_tool), None)
        if off_path is None:
            continue
        eligible += 1
        returns.extend((run_recovery(task, off_path, recover=False), run_recovery(task, off_path, recover=True)))
    variance = pvariance(returns) if len(returns) >= 2 else 0.0
    result = {
        "eligible_tasks": eligible,
        "counterfactual_returns": {"zero": returns.count(0.0), "one": returns.count(1.0)},
        "g0_variance": variance,
        "environment_support_gate_pass": eligible > 0 and variance > 0.0,
        "note": "deterministic environment-support gate; model Full Shadow still must pass its own Var(G0)>0 gate",
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["environment_support_gate_pass"]:
        raise SystemExit("controlled environment cannot express non-degenerate G0")


if __name__ == "__main__":
    main()
