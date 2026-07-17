#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from environments.api_bank.data import assign_splits, parse_api_catalog, parse_dialogue
from rescuecredit.logging import write_json


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def digest_records(records: list[dict]) -> str:
    payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for record in records)
    return hashlib.sha256(payload.encode()).hexdigest()


def cross_split_duplicates(splits: dict[str, list[dict]], key_fn) -> int:
    owners: dict[str, str] = {}
    duplicates = 0
    for split, records in splits.items():
        for record in records:
            key = key_fn(record)
            if key in owners and owners[key] != split:
                duplicates += 1
            else:
                owners[key] = split
    return duplicates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/DAMO-ConvAI/api-bank"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/api_bank_controlled_v1"))
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--full-shadow-size", type=int, default=30)
    args = parser.parse_args()

    catalog = parse_api_catalog(args.raw_root / "apis")
    sample_root = args.raw_root / "lv1-lv2-samples"
    tasks: list[dict] = []
    rejected = {"parse_failure": 0, "duplicate": 0}
    seen: set[tuple[str, str]] = set()
    source_files = sorted(sample_root.rglob("*.jsonl"))
    for path in source_files:
        task = parse_dialogue(path, catalog)
        if task is None:
            rejected["parse_failure"] += 1
            continue
        key = (task["normalized_goal_template"], task["reference_action_signature"])
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key)
        tasks.append(task)

    splits = assign_splits(tasks, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "api_catalog.json", catalog)
    selected_splits = {name: splits[name] for name in ("train", "dev", "test_id", "test_tool_ood")}
    all_tasks = [task for split in selected_splits for task in selected_splits[split]]
    write_jsonl(args.output_dir / "tasks_all.jsonl", all_tasks)
    for split, records in selected_splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", records)

    evaluation_pool = splits["dev"] + splits["test_id"]
    evaluation_pool = sorted(evaluation_pool, key=lambda task: hashlib.sha256(f"shadow:{task['task_id']}".encode()).hexdigest())
    full_shadow = evaluation_pool[: args.full_shadow_size]
    write_jsonl(args.output_dir / "full_shadow_eval.jsonl", full_shadow)

    source_commit = "unknown"
    repository_root = args.raw_root.parent
    git_dir = repository_root / ".git"
    head_path = git_dir / "HEAD"
    if head_path.exists():
        head = head_path.read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref_path = git_dir / head[5:]
            if ref_path.exists():
                source_commit = ref_path.read_text(encoding="utf-8").strip()
        elif len(head) == 40:
            source_commit = head
    train_families = {task["api_family_id"] for task in splits["train"]}
    ood_families = {task["api_family_id"] for task in splits["test_tool_ood"]}
    train_tools = {action["tool"] for task in splits["train"] for action in task["reference_actions"]}
    ood_tools = {action["tool"] for task in splits["test_tool_ood"] for action in task["reference_actions"]}
    manifest = {
        "name": "API-Bank-derived controlled environment",
        "not_official_leaderboard": True,
        "source_repository": "https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank",
        "source_commit": source_commit,
        "seed": args.seed,
        "official_raw_dialogues": len(source_files),
        "executable_after_filter": len(tasks),
        "train": len(splits["train"]),
        "dev": len(splits["dev"]),
        "test_id": len(splits["test_id"]),
        "test_tool_ood": len(splits["test_tool_ood"]),
        "split_conflict_excluded": len(splits["split_conflict_excluded"]),
        "full_shadow_eval": len(full_shadow),
        "api_families_train": len(train_families),
        "api_families_ood": len(ood_families),
        "rejected": rejected,
        "split_hashes": {split: digest_records(records) for split, records in selected_splits.items()},
        "leakage_checks": {
            "normalized_goal_cross_split": cross_split_duplicates(selected_splits, lambda task: task["normalized_goal_template"]),
            "reference_action_instance_cross_split": cross_split_duplicates(
                selected_splits, lambda task: json.dumps(task["reference_actions"], ensure_ascii=False, sort_keys=True)
            ),
            "ood_family_overlap_with_train": len(train_families & ood_families),
            "ood_atomic_tool_overlap_with_train": len(train_tools & ood_tools),
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
