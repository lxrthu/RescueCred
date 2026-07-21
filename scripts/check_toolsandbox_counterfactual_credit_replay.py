#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rescuecredit.counterfactual_credit_replay import replay_counterfactual_credit
from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from scripts.freeze_toolsandbox_counterfactual_credit_replay import STATUS


def _same_nested(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _same_nested(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_nested(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-10)
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--shadow-a-returns", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--behavior-ledger", type=Path, required=True)
    parser.add_argument("--replay-summary", type=Path, required=True)
    parser.add_argument("--replay-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    reported = json.loads(args.replay_summary.read_text(encoding="utf-8"))
    bank_manifest = json.loads(args.bank_manifest.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    predictions = torch.load(args.predictions, map_location="cpu", weights_only=True)
    artifact = torch.load(args.replay_artifact, map_location="cpu", weights_only=True)
    shadow_rows = read_jsonl(args.shadow_a_returns)
    shadow_by_id = {str(row["event_id"]): row for row in shadow_rows}
    event_ids = [str(value) for value in bank["event_ids"]]
    proposal_returns = torch.tensor(
        [
            float(shadow_by_id[event_id]["shadow_a_return"])
            if int(proposal_index) == 0
            else float(executed_return)
            for event_id, proposal_index, executed_return in zip(
                event_ids,
                bank["proposal_indices"],
                bank["executed_returns"],
                strict=True,
            )
        ],
        dtype=torch.float32,
    )
    config = protocol["config"]
    rebuilt, rebuilt_artifact = replay_counterfactual_credit(
        score_sketches=bank["score_sketches"],
        proposal_returns=proposal_returns,
        executed_returns=bank["executed_returns"],
        replaced=bank["replaced"],
        mean_predictions=predictions["mean_predictions"],
        groups=[str(value) for value in bank["task_ids"]],
        propensities=config["propensities"],
        primary_propensity=float(config["primary_propensity"]),
        replicates=int(config["replicates"]),
        task_bootstrap_replicates=int(config["task_bootstrap_replicates"]),
        seed=int(config["seed"]),
    )
    protocol_sha = file_sha256(args.protocol_lock)
    event_count = len(event_ids)
    tensor_lengths_valid = (
        all(
            len(bank[name]) == event_count
            for name in (
                "task_ids",
                "proposal_indices",
                "replaced",
                "score_norms",
                "executed_returns",
            )
        )
        and bank["score_sketches"].ndim == 2
        and len(bank["score_sketches"]) == event_count
        and all(
            len(predictions[name]) == event_count
            for name in ("mean_predictions", "scale_predictions", "fold_ids")
        )
    )
    finite_tensors = all(
        bool(torch.isfinite(tensor.float()).all())
        for tensor in (
            bank["score_sketches"],
            bank["score_norms"],
            bank["executed_returns"],
            predictions["mean_predictions"],
            predictions["scale_predictions"],
        )
    )
    integrity = {
        "protocol_status": protocol.get("status") == STATUS,
        "bank_bound": protocol.get("bank_sha256")
        == file_sha256(args.bank)
        == bank_manifest.get("bank_sha256"),
        "bank_manifest_bound": protocol.get("bank_manifest_sha256")
        == file_sha256(args.bank_manifest),
        "source_manifest_bound": protocol.get("source_manifest_sha256")
        == file_sha256(args.source_manifest)
        == bank_manifest.get("source_manifest_sha256"),
        "shadow_a_bound": protocol.get("shadow_a_sha256")
        == file_sha256(args.shadow_a_returns)
        == source_manifest.get("shadow_a_sha256"),
        "predictions_bound": protocol.get("prediction_sha256")
        == file_sha256(args.predictions)
        == reported.get("prediction_sha256")
        and predictions.get("bank_sha256") == file_sha256(args.bank)
        and predictions.get("source_manifest_sha256")
        == file_sha256(args.source_manifest),
        "behavior_ledger_bound": protocol.get("behavior_ledger_sha256")
        == file_sha256(args.behavior_ledger)
        == bank_manifest.get("behavior_ledger_sha256")
        == bank.get("behavior_ledger_sha256")
        == reported.get("behavior_ledger_sha256"),
        "behavior_sampled_before_outcome": bank_manifest.get(
            "behavior_sampled_before_private_outcomes"
        )
        is True,
        "old_proposal_not_relabelled": bank_manifest.get(
            "old_deepseek_proposal_reused_as_on_policy_sample"
        )
        is False,
        "primary_propensity_unclipped": bank_manifest.get(
            "primary_propensity_clipping"
        )
        is False,
        "bank_version": bank.get("version") == "rapg_candidate_policy_bank_v1",
        "replacement_identity": torch.equal(
            bank["replaced"].bool(), bank["proposal_indices"].long().eq(0)
        ),
        "proposal_replacement_encoding": bank["replaced"].dtype == torch.bool
        and bool(
            torch.isin(
                bank["proposal_indices"].long(),
                torch.tensor([0, 1], dtype=torch.long),
            ).all()
        ),
        "unique_event_ids": len(set(event_ids)) == event_count
        and len(shadow_by_id) == len(shadow_rows) == event_count
        and set(shadow_by_id) == set(event_ids),
        "tensor_shapes": tensor_lengths_valid,
        "finite_tensors": finite_tensors,
        "task_crossfit": bool(predictions.get("fold_audit"))
        and all(
            fold.get("task_overlap") == 0
            for fold in predictions.get("fold_audit", [])
        ),
        "source_scope_honest": source_manifest.get("role")
        == "development_surrogate_preflight_only"
        and source_manifest.get("outcome_direction_filter_used") is False
        and source_manifest.get("deployment_stream_representative") is False
        and source_manifest.get("replay_validity_conditioned") is True,
        "source_identity": bool(protocol.get("source_sha256"))
        and all(
            Path(path).is_file() and file_sha256(Path(path)) == expected
            for path, expected in protocol.get("source_sha256", {}).items()
        ),
        "reported_run_bound": reported.get("status") == "completed"
        and reported.get("protocol_lock_sha256") == protocol_sha
        and reported.get("artifact_sha256") == file_sha256(args.replay_artifact)
        and reported.get("bank_sha256") == file_sha256(args.bank)
        and reported.get("shadow_a_returns_sha256")
        == file_sha256(args.shadow_a_returns)
        and all(reported.get("integrity_checks", {}).values()),
        "metrics_recomputed": _same_nested(reported.get("metrics"), rebuilt),
        "artifact_recomputed": artifact.get("protocol_lock_sha256") == protocol_sha
        and artifact.get("bank_sha256") == file_sha256(args.bank)
        and all(
            torch.allclose(artifact[name], rebuilt_artifact[name], atol=1e-7, rtol=1e-7)
            for name in (
                "oracle_gradient",
                "naive_gradient",
                "firewall_gradient",
            )
        ),
        "analytic_aipw_identity": rebuilt.get("analytic_aipw_identity") is True,
        "historical_scope_honest": protocol.get(
            "historical_outcomes_previously_observed"
        )
        is True
        and reported.get("historical_outcomes_previously_observed") is True,
    }
    primary_tag = f"p{int(round(float(config['primary_propensity']) * 100)):02d}"
    naive = rebuilt["deterministic_methods"]["naive_b_to_a"]
    ipw = rebuilt["randomized_shadow"][primary_tag]["ipw"]
    aipw = rebuilt["randomized_shadow"][primary_tag]["aipw"]
    expected_audits = rebuilt["randomized_shadow"][primary_tag]["expected_audits"]
    tasks = rebuilt["task_results"]
    thresholds = protocol["thresholds"]
    aipw_mse_gain_over_ipw = (
        ipw["projected_gradient_mse"] - aipw["projected_gradient_mse"]
    ) / max(ipw["projected_gradient_mse"], 1e-12)
    checks = {
        "firewall_oracle_distance": rebuilt[
            "firewall_oracle_distance_ratio_vs_naive"
        ]
        <= thresholds["max_firewall_oracle_distance_ratio_vs_naive"],
        "firewall_task_improvement": tasks[
            "firewall_task_improvement_fraction_vs_naive"
        ]
        >= thresholds["min_firewall_task_improvement_fraction_vs_naive"],
        "rsc_aipw_mse_gain_over_ipw": aipw_mse_gain_over_ipw
        >= thresholds["min_rsc_aipw_mse_gain_over_ipw"],
        "rsc_aipw_cosine": aipw["mean_cosine_with_oracle"]
        > naive["cosine_with_oracle"],
        "rsc_aipw_task_improvement": tasks[
            "rsc_aipw_task_improvement_fraction_vs_naive"
        ]
        >= thresholds["min_rsc_aipw_task_improvement_fraction_vs_naive"],
        "firewall_task_concentration": tasks[
            "firewall_positive_improvement_top1_share"
        ]
        <= thresholds["max_positive_task_improvement_top1_share"],
        "rsc_aipw_task_concentration": tasks[
            "rsc_aipw_positive_improvement_top1_share"
        ]
        <= thresholds["max_positive_task_improvement_top1_share"],
        "firewall_task_bootstrap": tasks["bootstrap"]["firewall_vs_naive"][
            "lower95"
        ]
        > 0.0,
        "rsc_aipw_task_bootstrap": tasks["bootstrap"]["rsc_aipw_vs_naive"][
            "lower95"
        ]
        > 0.0,
        "matched_realized_audit_cost": abs(
            aipw["mean_realized_audits"] - expected_audits
        )
        <= thresholds["max_mean_audit_cost_error"],
    }
    passed = all(integrity.values()) and all(checks.values())
    result = {
        "passed": passed,
        "retrospective_credit_leakage_mechanism_supported": passed,
        "new_main_method_claim_supported": False,
        "paper_facing_policy_claim_supported": False,
        "stage": "toolsandbox_counterfactual_credit_replay_gate_seed42",
        "integrity_checks": integrity,
        "outcome_checks": checks,
        "observed": {
            "events": rebuilt["events"],
            "tasks": rebuilt["tasks"],
            "replacement_events": rebuilt["diagnostics"]["replacement_events"],
            "firewall_oracle_distance_ratio_vs_naive": rebuilt[
                "firewall_oracle_distance_ratio_vs_naive"
            ],
            "firewall_task_improvement_fraction_vs_naive": tasks[
                "firewall_task_improvement_fraction_vs_naive"
            ],
            "naive_cosine_with_oracle": naive["cosine_with_oracle"],
            "rsc_aipw_mean_cosine_with_oracle": aipw[
                "mean_cosine_with_oracle"
            ],
            "rsc_aipw_descriptive_monte_carlo_bias_relative": aipw[
                "descriptive_monte_carlo_bias_relative"
            ],
            "rsc_aipw_projected_gradient_mse": aipw[
                "projected_gradient_mse"
            ],
            "rsc_ipw_projected_gradient_mse": ipw["projected_gradient_mse"],
            "rsc_aipw_mse_gain_over_ipw": aipw_mse_gain_over_ipw,
            "rsc_aipw_task_improvement_fraction_vs_naive": tasks[
                "rsc_aipw_task_improvement_fraction_vs_naive"
            ],
            "firewall_positive_improvement_top1_share": tasks[
                "firewall_positive_improvement_top1_share"
            ],
            "rsc_aipw_positive_improvement_top1_share": tasks[
                "rsc_aipw_positive_improvement_top1_share"
            ],
            "firewall_task_bootstrap": tasks["bootstrap"]["firewall_vs_naive"],
            "rsc_aipw_task_bootstrap": tasks["bootstrap"][
                "rsc_aipw_vs_naive"
            ],
            "expected_audits": expected_audits,
            "mean_realized_audits": aipw["mean_realized_audits"],
            "credit_error_energy": rebuilt["diagnostics"],
        },
        "thresholds": thresholds,
        "protocol_lock_sha256": protocol_sha,
        "replay_summary_sha256": file_sha256(args.replay_summary),
        "claim_boundary": protocol["claim_boundary"],
        "next_step": (
            "use as a paper diagnostic figure, then test an actual policy update"
            if passed
            else "report the retrospective mechanism diagnostic as negative"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
