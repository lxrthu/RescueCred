#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pvariance

from environments.rescue_mdp.exact_solver import enumerate_q_values
from rescuecredit.estimators import residual_estimate
from rescuecredit.evaluation import mse, spearman
from rescuecredit.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/toy"))
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    q_records = enumerate_q_values()
    with (args.output_dir / "q_values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(q_records[0]))
        writer.writeheader()
        writer.writerows(q_records)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in q_records:
        groups[(record["state"], record["condition"])].append(record)
    mechanism = []
    for (state, condition), records in sorted(groups.items()):
        q0 = [record["q0"] for record in records]
        qh = [record["qh"] for record in records]
        rescue_scores = []
        for action_index, truth in enumerate(q0):
            rng = random.Random(args.seed + action_index)
            estimates = []
            for _ in range(2000):
                draw = int(rng.random() < 0.2)
                estimates.append(residual_estimate(0.5, draw, 0.2, truth if draw else None))
            rescue_scores.append(mean(estimates))
        mechanism.append(
            {
                "state": state,
                "condition": condition,
                "q0_variance": pvariance(q0),
                "group_reward_variance": pvariance(qh),
                "naive_gradient_alignment": spearman(q0, qh),
                "rescuecredit_gradient_alignment": spearman(q0, rescue_scores),
                "rescuecredit_scores_from_audit": rescue_scores,
                "pair_inversion_rate": mean(float((q0[i] - q0[j]) * (qh[i] - qh[j]) < 0) for i in range(len(q0)) for j in range(i + 1, len(q0))),
            }
        )
    write_json(args.output_dir / "mechanism.json", mechanism)

    estimator_records = []
    truth, mu = 0.2, 0.7
    for probability in (0.05, 0.10, 0.20, 0.40, 1.0):
        estimates = []
        rng = random.Random(args.seed)
        for _ in range(args.samples):
            draw = int(rng.random() < probability)
            estimates.append(residual_estimate(mu, draw, probability, truth if draw else None))
        empirical_variance = pvariance(estimates)
        theoretical_variance = (1.0 / probability - 1.0) * (truth - mu) ** 2
        estimator_records.append(
            {
                "p": probability,
                "samples": args.samples,
                "true_g0": truth,
                "mean": mean(estimates),
                "bias": mean(estimates) - truth,
                "mse": mse(estimates, [truth] * len(estimates)),
                "empirical_variance": empirical_variance,
                "theoretical_variance": theoretical_variance,
            }
        )
    write_json(args.output_dir / "estimator_validation.json", estimator_records)
    summary = {"not_research_evidence": False, "environment": "exact Rescue-MDP", "q_records": len(q_records), "estimator": estimator_records}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
