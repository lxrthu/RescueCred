#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.rapg_stability import summarize_task_stability


METHODS = ("uniform", "residual_only", "score_only", "rapg", "oracle")


def _markdown(audit: dict[str, Any]) -> str:
    stability = audit["stability"]
    bootstrap = stability["task_cluster_bootstrap"]
    loo = stability["leave_one_task_out"]
    cost = audit["matched_cost_audit"]
    rows = [
        "# RAPG frozen task-stability audit",
        "",
        "This is a fixed post-gate diagnostic. It does not reopen model selection, "
        "change the frozen gate, or support a policy-facing claim.",
        "",
        "## Raw comparison",
        "",
        "| Quantity | Uniform | RAPG | Difference / gain |",
        "|---|---:|---:|---:|",
        (
            "| Full-gradient design MSE | "
            f"{audit['raw_comparison']['uniform_design_mse']:.9g} | "
            f"{audit['raw_comparison']['rapg_design_mse']:.9g} | "
            f"{audit['raw_comparison']['mse_gain_over_uniform']:.2%} |"
        ),
        (
            "| Expected audits | "
            f"{cost['methods']['uniform']['expected_audits']:.9g} | "
            f"{cost['methods']['rapg']['expected_audits']:.9g} | "
            f"{cost['rapg_minus_uniform']:.3g} |"
        ),
        "",
        "## Stability findings",
        "",
        f"- Classification: `{stability['classification']}`.",
        (
            "- Task-cluster bootstrap 95% interval for MSE gain: "
            f"[{bootstrap['lower']:.2%}, {bootstrap['upper']:.2%}] "
            f"(median {bootstrap['median']:.2%})."
        ),
        (
            "- Leave-one-task-out gain: "
            f"minimum {loo['minimum']:.2%}, median {loo['median']:.2%}, "
            f"maximum {loo['maximum']:.2%}."
        ),
        (
            "- Improving tasks: "
            f"{stability['task_improvement_fraction']:.2%}; positive-improvement "
            "top-1 share: "
            f"{stability['positive_improvement_concentration']['top1_share']:.2%}."
        ),
        (
            "- Matched-cost discrepancy: "
            f"`{cost['classification']}` (maximum absolute error "
            f"{cost['maximum_absolute_error']:.3g})."
        ),
        "",
        "## Claim decision",
        "",
        f"- Frozen Pilot 0 gate remains failed: `{audit['frozen_gate_passed']}`.",
        f"- Paper-facing positive claim supported: `{audit['paper_positive_claim_supported']}`.",
        f"- Next experiment: {audit['next_step']}",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--shadow-a-returns", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--estimates", type=Path, required=True)
    parser.add_argument("--simulation-summary", type=Path, required=True)
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite RAPG stability audit")
    bank_manifest = json.loads(args.bank_manifest.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    simulation = json.loads(args.simulation_summary.read_text(encoding="utf-8"))
    pilot_gate = json.loads(args.pilot_gate.read_text(encoding="utf-8"))
    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    predictions = torch.load(args.predictions, map_location="cpu", weights_only=True)
    estimates = torch.load(args.estimates, map_location="cpu", weights_only=True)

    shadow_by_id = {
        str(row["event_id"]): row for row in read_jsonl(args.shadow_a_returns)
    }
    proposal_returns = torch.tensor(
        [
            float(shadow_by_id[str(event_id)]["shadow_a_return"])
            if int(proposal_index) == 0
            else float(executed_return)
            for event_id, proposal_index, executed_return in zip(
                bank["event_ids"],
                bank["proposal_indices"],
                bank["executed_returns"],
                strict=True,
            )
        ],
        dtype=torch.float32,
    )
    residuals = proposal_returns - predictions["mean_predictions"].float()
    replaced = bank["replaced"].bool()
    task_ids = [str(value) for value in bank["task_ids"]]
    unique_tasks = sorted(set(task_ids))

    task_rows: list[dict[str, Any]] = []
    for task_id in unique_tasks:
        indices = [index for index, value in enumerate(task_ids) if value == task_id]
        eligible = [index for index in indices if bool(replaced[index])]
        numerators: dict[str, float] = {}
        for method in ("uniform", "rapg"):
            probabilities = estimates["methods"][method]["probabilities"].float()
            numerators[method] = math.fsum(
                float(bank["score_norms"][index].square())
                * float(residuals[index].square())
                * (1.0 - float(probabilities[index]))
                / float(probabilities[index])
                for index in eligible
            )
        task_rows.append(
            {
                "task_id": task_id,
                "events": len(indices),
                "replacement_events": len(eligible),
                "uniform_numerator": numerators["uniform"],
                "rapg_numerator": numerators["rapg"],
                "uniform_task_mse": numerators["uniform"] / len(indices) ** 2,
                "rapg_task_mse": numerators["rapg"] / len(indices) ** 2,
                "task_mse_improvement": (
                    numerators["uniform"] - numerators["rapg"]
                )
                / len(indices) ** 2,
            }
        )

    stability = summarize_task_stability(
        task_rows,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        minimum_gain=0.15,
    )
    n = len(task_ids)
    audit_rate = float(simulation["audit_rate_per_all_events"])
    target_cost = audit_rate * n
    cost_methods: dict[str, dict[str, float]] = {}
    cost_errors: list[float] = []
    for method in METHODS:
        probabilities = estimates["methods"][method]["probabilities"].float()
        expected = math.fsum(float(value) for value in probabilities)
        error = expected - target_cost
        cost_methods[method] = {
            "expected_audits": expected,
            "target_audits": target_cost,
            "error": error,
            "absolute_error": abs(error),
        }
        cost_errors.append(abs(error))
    rapg_minus_uniform = (
        cost_methods["rapg"]["expected_audits"]
        - cost_methods["uniform"]["expected_audits"]
    )
    cost_errors.append(abs(rapg_minus_uniform))
    maximum_cost_error = max(cost_errors)
    matched_cost_classification = (
        "numerical_tolerance_only"
        if maximum_cost_error <= 1e-5
        else "substantive_budget_mismatch"
    )

    total_events = len(task_ids)
    total_uniform_numerator = math.fsum(
        row["uniform_numerator"] for row in task_rows
    )
    total_rapg_numerator = math.fsum(row["rapg_numerator"] for row in task_rows)
    recomputed_uniform_mse = total_uniform_numerator / total_events**2
    recomputed_rapg_mse = total_rapg_numerator / total_events**2
    gate_gain = float(pilot_gate["observed"]["mse_gain_over_uniform"])
    integrity = {
        "bank_bound": bank_manifest.get("bank_sha256") == file_sha256(args.bank),
        "source_manifest_bound": bank_manifest.get("source_manifest_sha256")
        == file_sha256(args.source_manifest),
        "shadow_returns_bound": source_manifest.get("shadow_a_sha256")
        == file_sha256(args.shadow_a_returns),
        "predictions_bound": simulation.get("prediction_sha256")
        == file_sha256(args.predictions),
        "estimates_bound": simulation.get("estimate_sha256")
        == file_sha256(args.estimates),
        "simulation_bound": simulation.get("bank_sha256") == file_sha256(args.bank),
        "frozen_gate_was_failed": pilot_gate.get("passed") is False,
        "event_and_task_counts_match": total_events == int(simulation["events"])
        and len(unique_tasks) == int(simulation["tasks"]),
        "uniform_design_mse_recomputed": math.isclose(
            recomputed_uniform_mse,
            float(simulation["methods"]["uniform"]["full_gradient_design_mse"]),
            abs_tol=1e-10,
        ),
        "rapg_design_mse_recomputed": math.isclose(
            recomputed_rapg_mse,
            float(simulation["methods"]["rapg"]["full_gradient_design_mse"]),
            abs_tol=1e-10,
        ),
        "gate_gain_recomputed": math.isclose(
            stability["observed_mse_gain_over_uniform"],
            gate_gain,
            abs_tol=1e-10,
        ),
        "analysis_inputs_frozen": True,
    }
    if not all(integrity.values()):
        raise RuntimeError({"integrity_failure": integrity})

    robust = stability["classification"] == (
        "surrogate_signal_robust_but_frozen_gate_failed"
    )
    next_step = (
        "freeze a fresh preregistered on-policy pilot; do not reuse this audit as its gate"
        if robust
        else "stop RAPG scaling; the apparent aggregate gain is not stable across tasks"
    )
    audit = {
        "status": "completed",
        "stage": "toolsandbox_rapg_frozen_task_stability_audit_seed42",
        "classification": stability["classification"],
        "frozen_gate_passed": False,
        "model_selection_reopened": False,
        "paper_positive_claim_supported": False,
        "integrity_checks": integrity,
        "raw_comparison": {
            "events": total_events,
            "tasks": len(unique_tasks),
            "uniform_design_mse": recomputed_uniform_mse,
            "rapg_design_mse": recomputed_rapg_mse,
            "mse_gain_over_uniform": stability["observed_mse_gain_over_uniform"],
        },
        "stability": stability,
        "matched_cost_audit": {
            "classification": matched_cost_classification,
            "tolerance": 1e-5,
            "maximum_absolute_error": maximum_cost_error,
            "rapg_minus_uniform": rapg_minus_uniform,
            "methods": cost_methods,
            "frozen_gate_value_changed": False,
        },
        "claim_boundary": (
            "fixed post-gate stability diagnostic on the replay-valid development "
            "surrogate; not an autoregressive RAPG, policy, or confirmatory result"
        ),
        "next_step": next_step,
        "artifact_hashes": {
            "bank_sha256": file_sha256(args.bank),
            "simulation_summary_sha256": file_sha256(args.simulation_summary),
            "pilot_gate_sha256": file_sha256(args.pilot_gate),
        },
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "statistical_audit.json", audit)
    write_jsonl(
        args.output_dir / "task_contributions.jsonl",
        sorted(task_rows, key=lambda row: row["task_mse_improvement"], reverse=True),
    )
    (args.output_dir / "STATISTICAL_AUDIT.md").write_text(
        _markdown(audit), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
