#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescuecredit.counterfactual_credit_replay import replay_counterfactual_credit
from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from scripts.freeze_toolsandbox_counterfactual_credit_replay import STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--shadow-a-returns", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--behavior-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch

    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite counterfactual replay output")
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    bank_manifest = json.loads(args.bank_manifest.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if protocol.get("status") != STATUS:
        raise ValueError("counterfactual replay protocol is not frozen")
    identity = {
        "bank": protocol.get("bank_sha256") == file_sha256(args.bank),
        "bank_manifest": protocol.get("bank_manifest_sha256")
        == file_sha256(args.bank_manifest),
        "source_manifest": protocol.get("source_manifest_sha256")
        == file_sha256(args.source_manifest),
        "predictions": protocol.get("prediction_sha256")
        == file_sha256(args.predictions),
        "behavior_ledger": protocol.get("behavior_ledger_sha256")
        == file_sha256(args.behavior_ledger),
        "shadow_a": protocol.get("shadow_a_sha256")
        == file_sha256(args.shadow_a_returns),
        "source_code": all(
            Path(path).is_file() and file_sha256(Path(path)) == expected
            for path, expected in protocol.get("source_sha256", {}).items()
        ),
    }
    if not all(identity.values()):
        raise ValueError({"counterfactual_replay_identity_failure": identity})
    if bank_manifest.get("bank_sha256") != protocol.get("bank_sha256"):
        raise ValueError("bank manifest/protocol mismatch")
    if source_manifest.get("shadow_a_sha256") != protocol.get("shadow_a_sha256"):
        raise ValueError("Shadow-A source/protocol mismatch")

    started = time.time()
    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    predictions = torch.load(args.predictions, map_location="cpu", weights_only=True)
    shadow_rows = read_jsonl(args.shadow_a_returns)
    shadow_by_id = {str(row["event_id"]): row for row in shadow_rows}
    if len(shadow_by_id) != len(shadow_rows):
        raise ValueError("Shadow-A returns contain duplicate event IDs")
    if set(shadow_by_id) != set(str(value) for value in bank["event_ids"]):
        raise ValueError("bank and Shadow-A event sets differ")
    if predictions.get("bank_sha256") != protocol.get("bank_sha256"):
        raise ValueError("cross-fit predictions are not bound to the bank")
    if predictions.get("source_manifest_sha256") != protocol.get(
        "source_manifest_sha256"
    ):
        raise ValueError("cross-fit predictions are not bound to the source manifest")
    if bank.get("version") != "rapg_candidate_policy_bank_v1":
        raise ValueError("unsupported RAPG bank version")
    if bank.get("behavior_ledger_sha256") != protocol.get(
        "behavior_ledger_sha256"
    ):
        raise ValueError("behavior ledger is not bound to the bank")
    if not torch.equal(
        bank["replaced"].bool(), bank["proposal_indices"].long().eq(0)
    ):
        raise ValueError("replacement identity is inconsistent")
    if bank["replaced"].dtype != torch.bool or not bool(
        torch.isin(
            bank["proposal_indices"].long(), torch.tensor([0, 1], dtype=torch.long)
        ).all()
    ):
        raise ValueError("proposal/replacement encoding is invalid")
    if not predictions.get("fold_audit") or not all(
        fold.get("task_overlap") == 0 for fold in predictions["fold_audit"]
    ):
        raise ValueError("predictions are not task-cross-fitted")
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
    config = protocol["config"]
    metrics, artifact = replay_counterfactual_credit(
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
    args.output_dir.mkdir(parents=True)
    artifact_path = args.output_dir / "replay_artifact.pt"
    torch.save(
        {
            **artifact,
            "bank_sha256": file_sha256(args.bank),
            "protocol_lock_sha256": file_sha256(args.protocol_lock),
        },
        artifact_path,
    )
    summary = {
        "status": "completed",
        "stage": "toolsandbox_counterfactual_credit_replay_seed42",
        "metrics": metrics,
        "integrity_checks": identity,
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "bank_sha256": file_sha256(args.bank),
        "bank_manifest_sha256": file_sha256(args.bank_manifest),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "shadow_a_returns_sha256": file_sha256(args.shadow_a_returns),
        "prediction_sha256": file_sha256(args.predictions),
        "behavior_ledger_sha256": file_sha256(args.behavior_ledger),
        "artifact_sha256": file_sha256(artifact_path),
        "historical_outcomes_previously_observed": True,
        "claim_boundary": protocol["claim_boundary"],
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "replay_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
