#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.rapg import (
    cross_fitted_residual_predictions,
    build_fixed_propensities,
    simulate_fixed_propensity_audits,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--shadow-a-returns", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--audit-rate", type=float, default=0.20)
    parser.add_argument("--p-min", type=float, default=0.05)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()

    import torch

    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite RAPG evaluation output")
    manifest = json.loads(args.bank_manifest.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("RAPG bank is not complete")
    if manifest.get("bank_sha256") != file_sha256(args.bank):
        raise ValueError("RAPG bank identity mismatch")
    if manifest.get("behavior_sampled_before_private_outcomes") is not True:
        raise ValueError("RAPG behavior/outcome ordering is not sealed")
    if manifest.get("old_deepseek_proposal_reused_as_on_policy_sample") is not False:
        raise ValueError("invalid RAPG behavior-policy identity")
    if manifest.get("primary_propensity_clipping") is not False:
        raise ValueError("primary RAPG bank enables propensity clipping")
    if source_manifest.get("shadow_a_sha256") != file_sha256(args.shadow_a_returns):
        raise ValueError("RAPG Shadow-A source identity mismatch")
    if source_manifest.get("deployment_stream_representative") is not False:
        raise ValueError("RAPG preflight source role changed unexpectedly")
    if manifest.get("source_manifest_sha256") != file_sha256(args.source_manifest):
        raise ValueError("RAPG bank/source manifest binding mismatch")

    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    if bank.get("version") != "rapg_candidate_policy_bank_v1":
        raise ValueError("unsupported RAPG bank version")
    if bank.get("behavior_ledger_sha256") != manifest.get(
        "behavior_ledger_sha256"
    ):
        raise ValueError("RAPG behavior ledger binding mismatch")
    started = time.time()
    shadow_rows = read_jsonl(args.shadow_a_returns)
    shadow_by_id = {str(row["event_id"]): row for row in shadow_rows}
    if set(shadow_by_id) != set(str(value) for value in bank["event_ids"]):
        raise ValueError("RAPG bank and Shadow-A event sets differ")
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
    means, scales, fold_ids, fold_audit = cross_fitted_residual_predictions(
        bank["public_features"],
        proposal_returns,
        [str(value) for value in bank["task_ids"]],
        folds=args.folds,
        seed=args.seed,
        ridge_alpha=args.ridge_alpha,
    )
    args.output_dir.mkdir(parents=True)
    primary_probabilities = build_fixed_propensities(
        score_norms=bank["score_norms"],
        replaced=bank["replaced"],
        scale_predictions=scales,
        audit_rate=args.audit_rate,
        p_min=args.p_min,
    )
    propensity_path = args.output_dir / "propensity_ledger.jsonl"
    write_jsonl(
        propensity_path,
        [
            {
                "event_id": str(event_id),
                "task_id_hash": str(task_id),
                "replaced": bool(replaced),
                "probabilities": {
                    method: float(values[index])
                    for method, values in primary_probabilities.items()
                },
                "shadow_a_opened_for_current_audit_draw": False,
            }
            for index, (event_id, task_id, replaced) in enumerate(
                zip(
                    bank["event_ids"],
                    bank["task_ids"],
                    bank["replaced"],
                    strict=True,
                )
            )
        ],
    )
    propensity_ledger_sha256 = file_sha256(propensity_path)
    oracle_probability = build_fixed_propensities(
        score_norms=bank["score_norms"],
        replaced=bank["replaced"],
        scale_predictions=scales,
        audit_rate=args.audit_rate,
        p_min=args.p_min,
        outcomes=proposal_returns,
        mean_predictions=means,
    )["oracle"]
    probabilities = {**primary_probabilities, "oracle": oracle_probability}
    simulation, estimates = simulate_fixed_propensity_audits(
        score_sketches=bank["score_sketches"],
        score_norms=bank["score_norms"],
        outcomes=proposal_returns,
        executed_returns=bank["executed_returns"],
        replaced=bank["replaced"],
        mean_predictions=means,
        scale_predictions=scales,
        groups=[str(value) for value in bank["task_ids"]],
        audit_rate=args.audit_rate,
        p_min=args.p_min,
        replicates=args.replicates,
        seed=args.seed + 700001,
        probabilities_by_method=probabilities,
    )
    prediction_path = args.output_dir / "crossfit_predictions.pt"
    estimate_path = args.output_dir / "audit_estimates.pt"
    torch.save(
        {
            "mean_predictions": means,
            "scale_predictions": scales,
            "fold_ids": fold_ids,
            "fold_audit": fold_audit,
            "bank_sha256": file_sha256(args.bank),
            "source_manifest_sha256": file_sha256(args.source_manifest),
            "propensity_ledger_sha256": propensity_ledger_sha256,
        },
        prediction_path,
    )
    torch.save(estimates, estimate_path)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_rapg_surrogate_preflight_simulation",
        **simulation,
        "seed": args.seed,
        "folds": args.folds,
        "ridge_alpha": args.ridge_alpha,
        "fold_audit": fold_audit,
        "bank_sha256": file_sha256(args.bank),
        "bank_manifest_sha256": file_sha256(args.bank_manifest),
        "prediction_sha256": file_sha256(prediction_path),
        "estimate_sha256": file_sha256(estimate_path),
        "propensity_ledger_sha256": propensity_ledger_sha256,
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "shadow_a_returns_sha256": file_sha256(args.shadow_a_returns),
        "ground_truth": manifest["ground_truth"],
        "behavior_policy_identity_bound": True,
        "deployment_visible_pre_audit_features": True,
        "propensities_committed_before_audit_resampling": True,
        "heldout_outcomes_physically_unopened_before_crossfit": False,
        "heldout_same_task_outcomes_used_for_propensity": False,
        "task_crossfit": True,
        "allocator_pretraining_shadow_cost_events": source_manifest[
            "shadow_source_cost_events"
        ],
        "fixed_shadow_cost": 1,
        "budget_semantics": "expected cost; no realized hard-cap truncation",
        "role": "candidate_selector_surrogate_preflight",
        "policy_pilot_authorized": False,
        "next_step_if_passed": "collect a clean on-policy autoregressive RAPG bank",
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "simulation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
