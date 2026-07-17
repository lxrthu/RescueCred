#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.azure_client import AzureOpenAIAdapter
from rescuecredit.logging import JsonlLogger
from rescuecredit.training import build_prompt, classify_patch, parse_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=Path("data/api_bank_controlled_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/azure/base_trajectories.jsonl"))
    args = parser.parse_args()
    tasks = [json.loads(line) for line in (args.data_dir / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][: args.limit]
    client = AzureOpenAIAdapter()
    logger = JsonlLogger(args.output)
    for task in tasks:
        expected = task["reference_actions"][0]
        prompt = build_prompt(task, [])
        completion = client.complete([{"role": "user", "content": prompt}], max_tokens=256, temperature=0.7)
        proposal = parse_action(completion)
        logger.write(
            {
                "task_id": task["task_id"],
                "source_sample_id": task["source_sample_id"],
                "completion": completion,
                "proposal_action": proposal,
                "patch_id": classify_patch(proposal, expected),
                "ground_truth_action": expected,
                "ground_truth_match": proposal == expected,
            }
        )
    print(f"saved {len(tasks)} Azure trajectories to {args.output}")


if __name__ == "__main__":
    main()
