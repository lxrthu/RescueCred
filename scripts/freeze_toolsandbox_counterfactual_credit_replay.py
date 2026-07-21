#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256
from rescuecredit.logging import write_json


STATUS = "frozen_before_counterfactual_credit_replay"
CONFIG = {
    "seed": 42,
    "replicates": 10_000,
    "task_bootstrap_replicates": 20_000,
    "propensities": [0.20, 0.30],
    "primary_propensity": 0.30,
}
THRESHOLDS = {
    "max_firewall_oracle_distance_ratio_vs_naive": 0.80,
    "min_firewall_task_improvement_fraction_vs_naive": 0.50,
    "min_rsc_aipw_mse_gain_over_ipw": 0.20,
    "require_rsc_aipw_cosine_above_naive": True,
    "min_rsc_aipw_task_improvement_fraction_vs_naive": 0.50,
    "max_positive_task_improvement_top1_share": 0.50,
    "require_task_bootstrap_lower95_above_zero": True,
    "max_mean_audit_cost_error": 0.10,
}
SOURCE_PATHS = (
    "rescuecredit/counterfactual_credit_replay.py",
    "scripts/freeze_toolsandbox_counterfactual_credit_replay.py",
    "scripts/run_toolsandbox_counterfactual_credit_replay.py",
    "scripts/check_toolsandbox_counterfactual_credit_replay.py",
    "scripts/cloud/run_toolsandbox_counterfactual_credit_replay_seed42.sh",
    "tests/test_counterfactual_credit_replay.py",
    "refine-logs/COUNTERFACTUAL_CREDIT_REPLAY_PLAN_20260721_204520.md",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--behavior-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    if args.output.exists():
        raise FileExistsError("refusing to replace counterfactual replay protocol")
    bank_manifest = json.loads(args.bank_manifest.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if bank_manifest.get("status") != "completed":
        raise ValueError("RAPG bank is incomplete")
    if bank_manifest.get("bank_sha256") != file_sha256(args.bank):
        raise ValueError("RAPG bank identity mismatch")
    if bank_manifest.get("source_manifest_sha256") != file_sha256(
        args.source_manifest
    ):
        raise ValueError("RAPG source lineage mismatch")
    if source_manifest.get("shadow_a_sha256") is None:
        raise ValueError("RAPG source does not seal Shadow-A outcomes")
    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    predictions = torch.load(args.predictions, map_location="cpu", weights_only=True)
    if bank.get("version") != "rapg_candidate_policy_bank_v1":
        raise ValueError("unsupported RAPG bank version")
    if bank_manifest.get("behavior_sampled_before_private_outcomes") is not True:
        raise ValueError("RAPG behavior/outcome ordering is not sealed")
    if bank_manifest.get("old_deepseek_proposal_reused_as_on_policy_sample") is not False:
        raise ValueError("invalid RAPG behavior-policy identity")
    if bank_manifest.get("primary_propensity_clipping") is not False:
        raise ValueError("RAPG primary propensity clipping must remain disabled")
    behavior_hash = file_sha256(args.behavior_ledger)
    if not (
        bank_manifest.get("behavior_ledger_sha256")
        == bank.get("behavior_ledger_sha256")
        == behavior_hash
    ):
        raise ValueError("RAPG behavior ledger binding mismatch")
    if not (
        source_manifest.get("role") == "development_surrogate_preflight_only"
        and source_manifest.get("outcome_direction_filter_used") is False
        and source_manifest.get("deployment_stream_representative") is False
        and source_manifest.get("replay_validity_conditioned") is True
    ):
        raise ValueError("RAPG source scope changed unexpectedly")
    event_count = len(bank["event_ids"])
    vector_names = (
        "proposal_indices",
        "replaced",
        "score_norms",
        "executed_returns",
    )
    if any(len(bank[name]) != event_count for name in vector_names) or len(
        bank["task_ids"]
    ) != event_count:
        raise ValueError("RAPG bank tensor lengths are inconsistent")
    if len(set(str(value) for value in bank["event_ids"])) != event_count:
        raise ValueError("RAPG bank contains duplicate event IDs")
    if bank["score_sketches"].ndim != 2 or len(bank["score_sketches"]) != event_count:
        raise ValueError("RAPG score sketches have invalid shape")
    if not torch.equal(
        bank["replaced"].bool(), bank["proposal_indices"].long().eq(0)
    ):
        raise ValueError("RAPG replacement identity is inconsistent")
    if bank["replaced"].dtype != torch.bool or not bool(
        torch.isin(
            bank["proposal_indices"].long(), torch.tensor([0, 1], dtype=torch.long)
        ).all()
    ):
        raise ValueError("RAPG proposal/replacement encoding is invalid")
    if not all(
        bool(torch.isfinite(bank[name].float()).all())
        for name in ("score_sketches", "score_norms", "executed_returns")
    ):
        raise ValueError("RAPG bank contains non-finite tensors")
    if predictions.get("bank_sha256") != file_sha256(args.bank) or predictions.get(
        "source_manifest_sha256"
    ) != file_sha256(args.source_manifest):
        raise ValueError("RAPG cross-fit predictions are not source-bound")
    if any(
        len(predictions[name]) != event_count
        for name in ("mean_predictions", "scale_predictions", "fold_ids")
    ):
        raise ValueError("RAPG cross-fit prediction lengths are inconsistent")
    if not all(
        bool(torch.isfinite(predictions[name].float()).all())
        for name in ("mean_predictions", "scale_predictions")
    ):
        raise ValueError("RAPG cross-fit predictions are non-finite")
    fold_audit = predictions.get("fold_audit", [])
    if not fold_audit or not all(
        fold.get("task_overlap") == 0 for fold in fold_audit
    ):
        raise ValueError("RAPG predictions are not task-cross-fitted")
    missing = [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError({"missing_counterfactual_replay_sources": missing})
    protocol = {
        "status": STATUS,
        "stage": "toolsandbox_counterfactual_credit_replay_seed42",
        "config": CONFIG,
        "thresholds": THRESHOLDS,
        "bank_sha256": file_sha256(args.bank),
        "bank_manifest_sha256": file_sha256(args.bank_manifest),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "prediction_sha256": file_sha256(args.predictions),
        "behavior_ledger_sha256": behavior_hash,
        "shadow_a_sha256": source_manifest["shadow_a_sha256"],
        "source_sha256": {
            path: file_sha256(Path(path)) for path in SOURCE_PATHS
        },
        "historical_outcomes_previously_observed": True,
        "claim_boundary": "retrospective credit-leakage diagnostic on the historical RAPG development surrogate; the AIPW arm repeats the prior uniform RAPG estimator and is not a new method, untouched confirmation, or policy-performance evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
