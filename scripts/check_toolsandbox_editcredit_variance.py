#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rescuecredit.edit_credit import gradient_noise_scale, minibatch_bootstrap_gradient_mse
from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash
from scripts.freeze_toolsandbox_editcredit_protocol import STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--full-sketches", type=Path, required=True)
    parser.add_argument("--edit-summary", type=Path, required=True)
    parser.add_argument("--edit-sketches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != STATUS:
        raise ValueError("invalid EditCredit protocol")
    train_rows = read_jsonl(args.train_file)
    truth = {str(row["event_id"]): row for row in train_rows}
    summaries = {
        "full_action": json.loads(args.full_summary.read_text(encoding="utf-8")),
        "editcredit": json.loads(args.edit_summary.read_text(encoding="utf-8")),
    }
    paths = {
        "full_action": args.full_sketches,
        "editcredit": args.edit_sketches,
    }
    rows_by_method = {method: read_jsonl(path) for method, path in paths.items()}
    integrity = {
        "protocol_status": True,
        "train_bank_bound": file_sha256(args.train_file) == protocol.get("train_sha256")
        and event_set_hash(train_rows) == protocol.get("train_event_set_hash"),
        "source_identity": bool(protocol.get("source_sha256"))
        and all(
            Path(path).is_file() and file_sha256(Path(path)) == expected
            for path, expected in protocol.get("source_sha256", {}).items()
        ),
        "same_frozen_events": True,
        "method_and_artifact_bound": True,
        "same_lora_initialization": summaries["full_action"].get(
            "initial_trainable_sha256"
        )
        == summaries["editcredit"].get("initial_trainable_sha256")
        and bool(summaries["full_action"].get("initial_trainable_sha256")),
        "countsketch_hash_bound": True,
        "task_groups_rebuilt": True,
        "finite_fixed_width_sketches": True,
    }
    expected_ids = set(truth)
    width = int(protocol["efficiency_config"]["gradient_sketch_buckets"])
    rng = random.Random(int(protocol["config"]["seed"]))
    prime = 2_147_483_647
    expected_coefficients = [
        rng.randrange(1, prime),
        rng.randrange(0, prime),
        rng.randrange(1, prime),
        rng.randrange(0, prime),
    ]
    if summaries["full_action"].get("countsketch_hash") != summaries["editcredit"].get(
        "countsketch_hash"
    ):
        integrity["countsketch_hash_bound"] = False
    for method, rows in rows_by_method.items():
        summary = summaries[method]
        ids = [str(row["event_id"]) for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            integrity["same_frozen_events"] = False
        if not (
            summary.get("status") == "completed"
            and summary.get("method") == method
            and summary.get("protocol_lock_sha256") == file_sha256(args.protocol_lock)
            and summary.get("train_file_sha256") == file_sha256(args.train_file)
            and summary.get("base_model_sha256") == protocol.get("base_model_sha256")
            and summary.get("sketches_sha256") == file_sha256(paths[method])
            and int(summary.get("buckets", -1)) == width
            and int(summary.get("seed", -1)) == int(protocol["config"]["seed"])
            and summary.get("event_set_hash") == protocol.get("train_event_set_hash")
            and summary.get("source_sha256") == protocol.get("source_sha256")
        ):
            integrity["method_and_artifact_bound"] = False
        if not (
            summary.get("countsketch_hash", {}).get("family")
            == "independent_affine_mod_prime"
            and int(summary.get("countsketch_hash", {}).get("prime", -1))
            == prime
            and summary.get("countsketch_hash", {}).get("coefficients")
            == expected_coefficients
        ):
            integrity["countsketch_hash_bound"] = False
        for row in rows:
            bound = truth.get(str(row["event_id"]))
            if (
                bound is None
                or str(row.get("task_id_hash")) != str(bound["task_id_hash"])
                or row.get("method") != method
                or row.get("decision") != bound.get("decision")
            ):
                integrity["task_groups_rebuilt"] = False
            try:
                if not float(row.get("gradient_norm", float("nan"))) >= 0.0:
                    integrity["finite_fixed_width_sketches"] = False
            except (TypeError, ValueError):
                integrity["finite_fixed_width_sketches"] = False
            sketch = row.get("sketch")
            if not isinstance(sketch, list) or len(sketch) != width:
                integrity["finite_fixed_width_sketches"] = False
                continue
            try:
                if not all(float(value) == float(value) and abs(float(value)) < float("inf") for value in sketch):
                    integrity["finite_fixed_width_sketches"] = False
            except (TypeError, ValueError):
                integrity["finite_fixed_width_sketches"] = False

    metrics = {}
    for method, rows in rows_by_method.items():
        ordered = sorted(rows, key=lambda row: str(row["event_id"]))
        sketches = [[float(value) for value in row["sketch"]] for row in ordered]
        metrics[method] = {
            **gradient_noise_scale(sketches),
            "minibatch_bootstrap": minibatch_bootstrap_gradient_mse(
                sketches,
                batch_size=int(
                    protocol["efficiency_config"]["gradient_bootstrap_batch_size"]
                ),
                replicates=int(protocol["efficiency_config"]["gradient_bootstrap_replicates"]),
                seed=int(protocol["config"]["seed"]) + 77,
            ),
            "wall_time_sec": float(summaries[method]["wall_time_sec"]),
            "forward_calls": int(summaries[method]["forward_calls"]),
        }
    noise_ratio = metrics["editcredit"]["gradient_noise_scale"] / max(
        metrics["full_action"]["gradient_noise_scale"], 1e-24
    )
    mse_ratio = metrics["editcredit"]["minibatch_bootstrap"]["mean_mse"] / max(
        metrics["full_action"]["minibatch_bootstrap"]["mean_mse"], 1e-24
    )
    checks = {
        "gradient_noise_scale_ratio": noise_ratio
        <= float(protocol["efficiency_config"]["max_gradient_noise_scale_ratio"]),
        "minibatch_gradient_mse_ratio": mse_ratio
        <= float(protocol["efficiency_config"]["max_minibatch_gradient_mse_ratio"]),
    }
    passed = all(integrity.values()) and all(checks.values())
    result = {
        "passed": passed,
        "stage": "toolsandbox_editcredit_gradient_variance_gate",
        "integrity_checks": integrity,
        "outcome_checks": checks,
        "methods": metrics,
        "observed": {
            "gradient_noise_scale_ratio": noise_ratio,
            "minibatch_gradient_mse_ratio": mse_ratio,
            "wall_time_ratio": metrics["editcredit"]["wall_time_sec"]
            / max(metrics["full_action"]["wall_time_sec"], 1e-12),
            "forward_call_ratio": metrics["editcredit"]["forward_calls"]
            / max(metrics["full_action"]["forward_calls"], 1),
        },
        "thresholds": protocol["efficiency_config"],
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "claim_boundary": "variance is compared between method-specific objectives and must not be presented as a same-estimand unbiased estimator result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
