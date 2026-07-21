#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.rapg import build_fixed_propensities


THRESHOLDS = {
    "min_mse_gain_over_uniform": 0.15,
    "min_mse_gain_over_residual_only": 0.0,
    "min_task_improvement_fraction": 0.50,
    "min_events": 100,
    "min_expected_ess": 5.0,
    "max_inverse_propensity": 20.0,
    "max_positive_task_improvement_share": 0.50,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--shadow-a-returns", type=Path, required=True)
    parser.add_argument("--simulation-summary", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--estimates", type=Path, required=True)
    parser.add_argument("--propensity-ledger", type=Path, required=True)
    parser.add_argument("--behavior-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    bank_manifest = json.loads(args.bank_manifest.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    summary = json.loads(args.simulation_summary.read_text(encoding="utf-8"))
    predictions = torch.load(args.predictions, map_location="cpu", weights_only=True)
    estimates = torch.load(args.estimates, map_location="cpu", weights_only=True)
    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    rapg = summary["methods"]["rapg"]

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
    recomputed_full = (
        bank["score_sketches"].float()
        * proposal_returns[:, None]
    ).mean(dim=0)
    bound_full = estimates["full_gradient"].float()
    propensity_rows = read_jsonl(args.propensity_ledger)
    if [str(row["event_id"]) for row in propensity_rows] != [
        str(value) for value in bank["event_ids"]
    ]:
        raise ValueError("propensity ledger event order mismatch")
    residual_vector = proposal_returns - predictions["mean_predictions"].float()
    eligible = torch.where(bank["replaced"].bool())[0].tolist()
    primary_probabilities = build_fixed_propensities(
        score_norms=bank["score_norms"],
        replaced=bank["replaced"],
        scale_predictions=predictions["scale_predictions"],
        audit_rate=0.20,
        p_min=0.05,
    )
    oracle_probability = build_fixed_propensities(
        score_norms=bank["score_norms"],
        replaced=bank["replaced"],
        scale_predictions=predictions["scale_predictions"],
        audit_rate=0.20,
        p_min=0.05,
        outcomes=proposal_returns,
        mean_predictions=predictions["mean_predictions"],
    )["oracle"]
    recomputed_probabilities = {
        **primary_probabilities,
        "oracle": oracle_probability,
    }
    recomputed_design_mse = {}
    recomputed_cosine = {}
    recomputed_expected_cost = {}
    propensity_matches = True
    estimates_match = True
    for method in ("uniform", "residual_only", "score_only", "rapg", "oracle"):
        probabilities = estimates["methods"][method]["probabilities"].float()
        propensity_matches = propensity_matches and torch.allclose(
            probabilities,
            recomputed_probabilities[method],
            atol=1e-7,
            rtol=1e-7,
        )
        if method != "oracle":
            ledger_probabilities = torch.tensor(
                [float(row["probabilities"][method]) for row in propensity_rows]
            )
            propensity_matches = propensity_matches and torch.allclose(
                probabilities, ledger_probabilities, atol=1e-7, rtol=1e-7
            )
        recomputed_design_mse[method] = sum(
            float(bank["score_norms"][index].square())
            * float(residual_vector[index].square())
            * (1.0 - float(probabilities[index]))
            / float(probabilities[index])
            for index in eligible
        ) / float(len(proposal_returns) ** 2)
        draws = estimates["methods"][method]["audit_draws"].bool()
        rebuilt_returns = bank["executed_returns"].float().repeat(len(draws), 1)
        rebuilt_returns[:, eligible] = predictions["mean_predictions"][eligible][
            None, :
        ] + (
            draws.float() / recomputed_probabilities[method][eligible][None, :]
        ) * (
            proposal_returns[eligible]
            - predictions["mean_predictions"][eligible]
        )[None, :]
        rebuilt_estimates = torch.einsum(
            "rn,nd->rd", rebuilt_returns, bank["score_sketches"].float()
        ) / float(len(proposal_returns))
        method_estimates = estimates["methods"][method]["gradient_estimates"].float()
        estimates_match = estimates_match and torch.allclose(
            rebuilt_estimates, method_estimates, atol=1e-6, rtol=1e-6
        )
        recomputed_cosine[method] = float(
            (
                method_estimates
                @ recomputed_full
                / (
                    method_estimates.norm(dim=1).clamp_min(1e-12)
                    * recomputed_full.norm().clamp_min(1e-12)
                )
            ).mean()
        )
        recomputed_expected_cost[method] = float(probabilities.sum())
    uniform_mse = recomputed_design_mse["uniform"]
    residual_mse = recomputed_design_mse["residual_only"]
    rapg_mse = recomputed_design_mse["rapg"]
    gain_uniform = (uniform_mse - rapg_mse) / max(uniform_mse, 1e-12)
    gain_residual = (residual_mse - rapg_mse) / max(residual_mse, 1e-12)
    groups = [str(value) for value in bank["task_ids"]]
    unique_groups = sorted(set(groups))
    task_improvements = []
    for group in unique_groups:
        indices = [index for index, value in enumerate(groups) if value == group]
        group_eligible = [index for index in indices if index in eligible]
        method_task_mse = {}
        for method in ("uniform", "rapg"):
            probabilities = estimates["methods"][method]["probabilities"].float()
            method_task_mse[method] = sum(
                float(bank["score_norms"][index].square())
                * float(residual_vector[index].square())
                * (1.0 - float(probabilities[index]))
                / float(probabilities[index])
                for index in group_eligible
            ) / float(len(indices) ** 2)
        task_improvements.append(
            method_task_mse["uniform"] - method_task_mse["rapg"]
        )
    task_improvement_fraction = sum(value > 0 for value in task_improvements) / len(
        task_improvements
    )
    positive_task_improvements = [max(0.0, value) for value in task_improvements]
    positive_total = sum(positive_task_improvements)
    max_task_share = (
        max(positive_task_improvements, default=0.0) / positive_total
        if positive_total > 0
        else 1.0
    )
    rapg_probabilities = estimates["methods"]["rapg"]["probabilities"].float()
    selected_rapg_probabilities = [float(rapg_probabilities[index]) for index in eligible]
    expected_ess = len(eligible) ** 2 / sum(
        1.0 / probability for probability in selected_rapg_probabilities
    )
    max_inverse_propensity = max(
        1.0 / probability for probability in selected_rapg_probabilities
    )
    analytic_ht_identity = True
    for method in ("uniform", "residual_only", "score_only", "rapg"):
        probabilities = estimates["methods"][method]["probabilities"].float()
        analytic_expectation = bank["executed_returns"].float().clone()
        analytic_expectation[eligible] = predictions["mean_predictions"][eligible] + (
            probabilities[eligible] / probabilities[eligible]
        ) * (proposal_returns[eligible] - predictions["mean_predictions"][eligible])
        analytic_ht_identity = analytic_ht_identity and torch.allclose(
            analytic_expectation, proposal_returns, atol=1e-7, rtol=1e-7
        )
    integrity = {
        "bank_bound": bank_manifest.get("bank_sha256") == file_sha256(args.bank),
        "summary_bound": summary.get("bank_sha256") == file_sha256(args.bank),
        "bank_manifest_bound": summary.get("bank_manifest_sha256")
        == file_sha256(args.bank_manifest),
        "source_manifest_bound": summary.get("source_manifest_sha256")
        == file_sha256(args.source_manifest)
        and bank_manifest.get("source_manifest_sha256")
        == file_sha256(args.source_manifest),
        "source_files_bound": source_manifest.get("shadow_a_sha256")
        == file_sha256(args.shadow_a_returns),
        "predictions_bound": summary.get("prediction_sha256")
        == file_sha256(args.predictions),
        "estimates_bound": summary.get("estimate_sha256")
        == file_sha256(args.estimates),
        "propensity_ledger_bound": summary.get("propensity_ledger_sha256")
        == file_sha256(args.propensity_ledger)
        and predictions.get("propensity_ledger_sha256")
        == file_sha256(args.propensity_ledger),
        "propensities_match_ledger": bool(propensity_matches),
        "behavior_ledger_bound": bank_manifest.get("behavior_ledger_sha256")
        == file_sha256(args.behavior_ledger)
        and bank.get("behavior_ledger_sha256")
        == file_sha256(args.behavior_ledger),
        "audit_estimates_rebuilt": bool(estimates_match),
        "behavior_sampled_before_outcome": bank_manifest.get(
            "behavior_sampled_before_private_outcomes"
        )
        is True,
        "old_deepseek_proposal_not_relabelled": bank_manifest.get(
            "old_deepseek_proposal_reused_as_on_policy_sample"
        )
        is False,
        "behavior_policy_identity": summary.get("behavior_policy_identity_bound")
        is True,
        "task_crossfit": summary.get("task_crossfit") is True
        and all(fold.get("task_overlap") == 0 for fold in summary["fold_audit"]),
        "deployable_pre_audit_features": summary.get(
            "deployment_visible_pre_audit_features"
        )
        is True,
        "offline_crossfit_scope_honest": summary.get(
            "heldout_outcomes_physically_unopened_before_crossfit"
        )
        is False
        and summary.get("heldout_same_task_outcomes_used_for_propensity") is False
        and summary.get("policy_pilot_authorized") is False,
        "source_scope_honest": source_manifest.get("role")
        == "development_surrogate_preflight_only"
        and source_manifest.get("outcome_direction_filter_used") is False
        and source_manifest.get("deployment_stream_representative") is False
        and source_manifest.get("replay_validity_conditioned") is True,
        "allocator_source_cost_reported": int(
            summary.get("allocator_pretraining_shadow_cost_events", -1)
        )
        == int(source_manifest.get("shadow_source_cost_events", -2)),
        "fixed_cost": summary.get("fixed_shadow_cost") == 1,
        "no_primary_weight_clipping": summary.get("primary_weights_clipped")
        is False,
        "full_gradient_recomputed": torch.allclose(
            recomputed_full, bound_full, atol=1e-6, rtol=1e-6
        ),
        "all_design_mse_recomputed": all(
            abs(
                recomputed_design_mse[method]
                - float(summary["methods"][method]["full_gradient_design_mse"])
            )
            <= 1e-8
            for method in recomputed_design_mse
        ),
        "summary_effects_recomputed": abs(
            gain_uniform - float(summary["rapg_mse_gain_over_uniform"])
        )
        <= 1e-8
        and abs(
            gain_residual - float(summary["rapg_mse_gain_over_residual_only"])
        )
        <= 1e-8
        and abs(
            task_improvement_fraction - float(summary["task_improvement_fraction"])
        )
        <= 1e-8
        and abs(
            max_task_share
            - float(summary["max_positive_task_improvement_share"])
        )
        <= 1e-8,
        "prediction_bank_bound": predictions.get("bank_sha256")
        == file_sha256(args.bank),
    }
    outcomes = {
        "minimum_event_count": len(bank["event_ids"])
        == int(summary["events"])
        == int(bank_manifest["events"])
        == int(source_manifest["events"])
        and len(bank["event_ids"]) >= THRESHOLDS["min_events"],
        "analytic_ht_identity": bool(analytic_ht_identity),
        "mse_gain_over_uniform": gain_uniform
        >= THRESHOLDS["min_mse_gain_over_uniform"],
        "mse_gain_over_residual_only": gain_residual
        > THRESHOLDS["min_mse_gain_over_residual_only"],
        "cosine_beats_uniform": recomputed_cosine["rapg"]
        > recomputed_cosine["uniform"],
        "task_improvement": task_improvement_fraction
        >= THRESHOLDS["min_task_improvement_fraction"],
        "task_concentration": max_task_share
        <= THRESHOLDS["max_positive_task_improvement_share"],
        "ess": expected_ess >= THRESHOLDS["min_expected_ess"],
        "max_weight": max_inverse_propensity
        <= THRESHOLDS["max_inverse_propensity"],
        "matched_expected_cost": abs(
            recomputed_expected_cost["rapg"]
            - recomputed_expected_cost["uniform"]
        )
        <= 1e-6
        and abs(
            recomputed_expected_cost["rapg"]
            - recomputed_expected_cost["residual_only"]
        )
        <= 1e-6
        and abs(recomputed_expected_cost["rapg"] - 0.20 * len(bank["event_ids"]))
        <= 1e-6,
    }
    passed = all(integrity.values()) and all(outcomes.values())
    payload = {
        "status": "passed" if passed else "failed",
        "stage": "toolsandbox_rapg_surrogate_preflight_gate",
        "passed": passed,
        "integrity_checks": integrity,
        "outcome_checks": outcomes,
        "thresholds": THRESHOLDS,
        "observed": {
            "projected_bias_relative_secondary": rapg[
                "projected_bias_relative"
            ],
            "rapg_full_gradient_design_mse": rapg_mse,
            "uniform_full_gradient_design_mse": uniform_mse,
            "residual_only_full_gradient_design_mse": residual_mse,
            "mse_gain_over_uniform": gain_uniform,
            "mse_gain_over_residual_only": gain_residual,
            "task_improvement_fraction": task_improvement_fraction,
            "rapg_mean_cosine": recomputed_cosine["rapg"],
            "uniform_mean_cosine": recomputed_cosine["uniform"],
            "max_inverse_propensity": max_inverse_propensity,
            "expected_ess": expected_ess,
        },
        "claim_boundary": (
            "candidate-selector surrogate preflight on a replay-valid development stress set; not an autoregressive RAPG or policy result"
        ),
        "next_step": (
            "collect a clean on-policy autoregressive RAPG bank"
            if passed
            else "stop before clean on-policy collection; report failed surrogate preflight"
        ),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
