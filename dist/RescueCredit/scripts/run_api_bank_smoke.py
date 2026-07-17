#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from environments.api_bank import APIBankControlledEnv, APIBankHarness
from rescuecredit.logging import JsonlLogger, write_json


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corrupted(reference: dict, patch: str, fallback_tool: str) -> dict:
    if patch == "premature_finish":
        return {"type": "finish"}
    action = copy.deepcopy(reference)
    if patch == "missing_required_argument":
        if action["arguments"]:
            action["arguments"].pop(sorted(action["arguments"])[0])
    else:
        action["tool"] = fallback_tool if fallback_tool != reference["tool"] else "DefinitelyWrongTool"
    return action


def run_episode(task: dict, condition: str, patch: str, seed: int) -> dict:
    env = APIBankControlledEnv(task["max_steps"])
    harness = APIBankHarness(condition)
    env.reset(task, seed)
    state_ref = env.snapshot()
    rng_state = env.get_rng_state()
    replay_hash = env.observation()["state_hash"]
    first = True
    interventions = 0
    while not env.done:
        expected = env.expected_action()
        if expected is None:
            proposal = {"type": "finish"}
        elif first:
            fallback = task["available_tools"][-1]["name"] if task["available_tools"] else "WrongTool"
            proposal = corrupted(expected, patch, fallback)
        else:
            proposal = expected
        executed, decision = harness.execute(env.observation(), proposal, expected)
        interventions += int(decision.triggered and decision.patch_id != "placebo")
        _, reward, done, _ = env.step(executed)
        first = False
        if done:
            assisted_success = bool(reward)
    main_steps = env.steps
    env.restore(state_ref)
    replay_valid = env.observation()["state_hash"] == replay_hash and env.get_rng_state() == rng_state
    return {
        "task_id": task["task_id"],
        "condition": condition,
        "patch": patch,
        "assisted_success": assisted_success,
        "first_pass_valid": patch == "none",
        "num_teachable_interventions": interventions,
        "main_env_steps": main_steps,
        "shadow_env_steps": 0,
        "replay_valid": replay_valid,
        "not_research_evidence": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/api_bank_controlled_v1"))
    parser.add_argument("--split", choices=["train", "dev", "test_id", "test_tool_ood"], default="dev")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke/api_bank"))
    args = parser.parse_args()
    tasks = read_jsonl(args.data_dir / f"{args.split}.jsonl")[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(args.output_dir / "trajectory.jsonl")
    records = []
    for task_index, task in enumerate(tasks):
        for patch in ("missing_required_argument", "wrong_tool_replace", "premature_finish"):
            for condition in ("H0", "H3"):
                record = run_episode(task, condition, patch, 20260714 + task_index)
                logger.write(record)
                records.append(record)
    summary = {
        "not_research_evidence": True,
        "reason": "mechanically injected errors for infrastructure validation",
        "tasks": len(tasks),
        "episodes": len(records),
        "replay_failure_rate": sum(not item["replay_valid"] for item in records) / len(records) if records else None,
        "h0_success_rate": sum(item["assisted_success"] for item in records if item["condition"] == "H0") / max(1, sum(item["condition"] == "H0" for item in records)),
        "h3_success_rate": sum(item["assisted_success"] for item in records if item["condition"] == "H3") / max(1, sum(item["condition"] == "H3" for item in records)),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
