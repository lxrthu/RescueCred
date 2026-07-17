#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ["s_on", "s_off", "dependence_gap", "first_pass", "intervention_rate"]


def bootstrap_ci(values: list[float], samples: int, seed: int = 20260714) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def read_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task_id = record["task_id"]
        if task_id in records:
            raise SystemExit(f"duplicate task_id in {path}: {task_id}")
        records[task_id] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--expected-seeds", default="42,43,44")
    parser.add_argument("--budget-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    expected_seeds = {int(value) for value in args.expected_seeds.split(",") if value}
    records = []
    for path in args.inputs:
        summary = json.loads(path.read_text(encoding="utf-8"))
        run_summary_path = path.parent.parent / "run_summary.json"
        if not run_summary_path.exists():
            raise SystemExit(f"missing run summary for {path}: {run_summary_path}")
        run = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if summary["method"] != run["method"] or int(summary["seed"]) != int(run["seed"]):
            raise SystemExit(f"eval/run method or seed mismatch: {path}")
        task_results_path = path.parent / "task_results.jsonl"
        if not task_results_path.exists():
            raise SystemExit(f"missing task-level results: {task_results_path}")
        records.append({"summary": summary, "run": run, "tasks": read_jsonl(task_results_path), "path": path})

    comparison_contracts = {
        json.dumps(
            {
                "model": item["run"]["model"],
                "model_revision": item["run"]["model_revision"],
                "train_split_hash": item["run"]["split_hash"],
                "comparability": item["run"]["comparability"],
            },
            sort_keys=True,
        )
        for item in records
    }
    if len(comparison_contracts) != 1:
        raise SystemExit("model/revision/train split/hyperparameter comparability check failed")
    for split in {item["summary"]["split"] for item in records}:
        split_hashes = {item["summary"]["split_hash"] for item in records if item["summary"]["split"] == split}
        if len(split_hashes) != 1:
            raise SystemExit(f"cross-method eval split hash mismatch for {split}")
    budgets = [int(item["run"]["total_training_steps"]) for item in records]
    target = max(budgets) if budgets else 0
    if target and (target - min(budgets)) / target > args.budget_tolerance:
        raise SystemExit(f"equal-interaction check failed: min={min(budgets)} max={target}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_method_split_seed = {}
    for item in records:
        summary = item["summary"]
        grouped[(summary["method"], summary["split"])].append(summary)
        by_method_split_seed[(summary["method"], summary["split"], int(summary["seed"]))] = item
    rows = []
    for (method, split), items in sorted(grouped.items()):
        seeds = {int(item["seed"]) for item in items}
        if seeds != expected_seeds:
            raise SystemExit(f"seed completeness failed for {method}/{split}: {sorted(seeds)}")
        split_hashes = {item["split_hash"] for item in items}
        if len(split_hashes) != 1:
            raise SystemExit(f"split hash mismatch for {method}/{split}")
        row = {"method": method, "split": split, "seeds": len(items)}
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            low, high = bootstrap_ci(values, args.bootstrap_samples)
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)

    paired = []
    for split in sorted({item["summary"]["split"] for item in records}):
        for baseline in ("mask_correction", "naive_h_grpo"):
            for metric in ("s_off", "first_pass"):
                deltas_by_task: dict[str, list[float]] = defaultdict(list)
                for seed in sorted(expected_seeds):
                    try:
                        rescue = by_method_split_seed[("rescuecredit", split, seed)]["tasks"]
                        other = by_method_split_seed[(baseline, split, seed)]["tasks"]
                    except KeyError as error:
                        raise SystemExit(f"missing paired method/split/seed record: {error}") from error
                    task_ids = sorted(set(rescue) & set(other))
                    if len(task_ids) != len(rescue) or len(task_ids) != len(other):
                        raise SystemExit(f"paired task coverage mismatch for {baseline}/{split}/seed{seed}")
                    for task_id in task_ids:
                        deltas_by_task[task_id].append(float(rescue[task_id][metric]) - float(other[task_id][metric]))
                incomplete = [task_id for task_id, values in deltas_by_task.items() if len(values) != len(expected_seeds)]
                if incomplete:
                    raise SystemExit(f"incomplete paired seeds for {baseline}/{split}: {incomplete[:3]}")
                task_deltas = [mean(values) for values in deltas_by_task.values()]
                low, high = bootstrap_ci(task_deltas, args.bootstrap_samples)
                paired.append(
                    {
                        "comparison": f"rescuecredit-minus-{baseline}",
                        "split": split,
                        "metric": metric,
                        "paired_tasks": len(task_deltas),
                        "seeds_averaged_per_task": len(expected_seeds),
                        "mean_delta": mean(task_deltas),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "main_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["method", "split"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "main_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "paired_comparisons.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = ["| Method | Split | Seeds | S_on | S_off | First-pass | IR |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        markdown.append(
            f"| {row['method']} | {row['split']} | {row['seeds']} | {row['s_on_mean']:.4f} | "
            f"{row['s_off_mean']:.4f} | {row['first_pass_mean']:.4f} | {row['intervention_rate_mean']:.4f} |"
        )
    (args.output_dir / "main_results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"aggregated {len(records)} eval files with paired task-level bootstrap into {args.output_dir}")


if __name__ == "__main__":
    main()
