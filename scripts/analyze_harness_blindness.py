#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import directory_sha256, file_sha256
from rescuecredit.logging import write_json


METHODS = ("naive_h_grpo", "mask_correction", "rescuecredit")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_evaluation_artifacts(eval_dir: Path, summary: dict[str, Any]) -> None:
    declared = summary.get("evaluation_artifacts", [])
    expected_names = {"trajectory.jsonl", "task_results.jsonl"}
    if {entry.get("path") for entry in declared} != expected_names:
        raise ValueError(f"evaluation artifact set mismatch in {eval_dir}")
    for entry in declared:
        path = eval_dir / entry["path"]
        if not path.is_file():
            raise ValueError(f"missing evaluation artifact: {path}")
        rows = sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if file_sha256(path) != entry.get("sha256") or rows != entry.get("rows"):
            raise ValueError(f"evaluation artifact integrity mismatch: {path}")
    task_rows = read_jsonl(eval_dir / "task_results.jsonl")
    count = max(1, len(task_rows))
    margins = [
        float(value)
        for row in task_rows
        for value in row.get("preference_margins_b_over_a", [])
    ]
    recomputed = {
        "num_tasks": len(task_rows),
        "s_on": sum(float(row["s_on"]) for row in task_rows) / count,
        "s_off": sum(float(row["s_off"]) for row in task_rows) / count,
        "first_attempt_accuracy": sum(
            float(row["first_attempt_accuracy"]) for row in task_rows
        )
        / count,
        "intervention_rate": sum(bool(row["intervened"]) for row in task_rows)
        / count,
        "rescued_tasks": sum(
            row["s_on"] == 1.0 and row["s_off"] == 0.0 for row in task_rows
        ),
        "harmed_tasks": sum(
            row["s_on"] == 0.0 and row["s_off"] == 1.0 for row in task_rows
        ),
        "evaluation_steps": sum(
            int(row["harness_on_steps"]) + int(row["harness_off_steps"])
            for row in task_rows
        ),
        "preference_margin_events": len(margins),
        "mean_preference_margin_b_over_a": (
            sum(margins) / len(margins) if margins else None
        ),
    }
    recomputed["dependence_gap"] = recomputed["s_on"] - recomputed["s_off"]
    for key, value in recomputed.items():
        claimed = summary.get(key)
        if value is None or claimed is None:
            matches = value is claimed
        elif isinstance(value, float):
            matches = math.isclose(
                float(claimed), value, rel_tol=1e-9, abs_tol=1e-12
            )
        else:
            matches = claimed == value
        if not matches:
            raise ValueError(
                f"evaluation metric mismatch in {eval_dir}: {key}={claimed!r}, expected {value!r}"
            )


def training_signal(
    root: Path,
    *,
    expected_method: str | None = None,
    expected_config_hash: str | None = None,
    expected_run_id: str | None = None,
    expected_lambda_corr: float | None = None,
) -> dict[str, Any]:
    rescued_prefix: list[float] = []
    harmed_prefix: list[float] = []
    zero_prefix: list[float] = []
    correction_updates = 0
    shadow_alignment_checks = 0
    shadow_alignment_matches = 0
    train_rows = 0
    realized_group_sizes: set[int] = set()
    seen_rows: set[tuple[int, int]] = set()
    for path in sorted(root.glob("train_rank*.jsonl")):
        for row in read_jsonl(path):
            if expected_method is not None and row.get("method") != expected_method:
                raise ValueError(f"foreign method row in {path}")
            if expected_config_hash is not None and row.get(
                "config_hash"
            ) != expected_config_hash:
                raise ValueError(f"foreign config row in {path}")
            if expected_run_id is not None and row.get("run_id") != expected_run_id:
                raise ValueError(f"foreign run row in {path}")
            identity = (int(row.get("rank", -1)), int(row.get("update", -1)))
            if identity in seen_rows:
                raise ValueError(f"duplicate training row identity {identity}")
            seen_rows.add(identity)
            train_rows += 1
            realized_group_sizes.add(int(row.get("realized_group_size", -1)))
            raw_correction = float(row.get("loss_corr", float("nan")))
            weighted_correction = float(
                row.get("weighted_loss_corr", float("nan"))
            )
            if not math.isfinite(raw_correction) or not math.isfinite(
                weighted_correction
            ):
                raise ValueError(f"non-finite correction loss in {path}")
            if expected_lambda_corr is not None and not math.isclose(
                weighted_correction,
                expected_lambda_corr * raw_correction,
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise ValueError(f"weighted correction loss mismatch in {path}")
            correction_updates += int(
                abs(weighted_correction) > 1e-12
            )
            assisted = row.get("assisted_returns", [])
            shadows = row.get("diagnostic_shadow_returns", [])
            replay = row.get("diagnostic_replay_valid", [])
            assigned = row.get("prefix_assigned_advantages", [])
            g0_hats = row.get("g0_hat", [])
            if not (
                len(assisted)
                == len(shadows)
                == len(replay)
                == len(assigned)
                == len(g0_hats)
            ):
                raise ValueError(f"diagnostic arrays are misaligned in {path}")
            for gh, g0, valid, advantage, g0_hat in zip(
                assisted, shadows, replay, assigned, g0_hats
            ):
                if valid is not True or g0 is None or advantage is None:
                    continue
                shadow_alignment_checks += 1
                shadow_alignment_matches += int(
                    abs(float(g0_hat) - float(g0)) <= 1e-12
                )
                delta = float(gh) - float(g0)
                target = (
                    rescued_prefix
                    if delta > 1e-12
                    else harmed_prefix
                    if delta < -1e-12
                    else zero_prefix
                )
                target.append(float(advantage))

    def summarize(values: list[float]) -> dict[str, Any]:
        return {
            "events": len(values),
            "mean_prefix_assigned_advantage": (
                sum(values) / len(values) if values else None
            ),
            "negative_prefix_rate": (
                sum(value < -1e-12 for value in values) / len(values)
                if values
                else None
            ),
            "zero_prefix_rate": (
                sum(abs(value) <= 1e-12 for value in values) / len(values)
                if values
                else None
            ),
            "positive_prefix_rate": (
                sum(value > 1e-12 for value in values) / len(values)
                if values
                else None
            ),
        }

    return {
        "rescued": summarize(rescued_prefix),
        "harmed": summarize(harmed_prefix),
        "zero_delta": summarize(zero_prefix),
        "nonzero_correction_updates": correction_updates,
        "shadow_alignment_checks": shadow_alignment_checks,
        "shadow_alignment_rate": (
            shadow_alignment_matches / shadow_alignment_checks
            if shadow_alignment_checks
            else None
        ),
        "train_rows": train_rows,
        "realized_group_sizes": sorted(realized_group_sizes),
    }


def checkpoint_curve(
    root: Path,
    *,
    method: str | None = None,
    expected: dict[str, Any] | None = None,
    dev_split_hash: str | None = None,
    manifest_sha256: str | None = None,
    dev_sha256: str | None = None,
    protocol_lock_sha256: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("eval_*/eval_summary.json")):
        summary = read_json(path)
        verify_evaluation_artifacts(path.parent, summary)
        checkpoint_name = path.parent.name.removeprefix("eval_")
        if expected is not None:
            expected_checkpoint = root / "checkpoints" / checkpoint_name
            checks = {
                "method": summary.get("method") == method,
                "seed": summary.get("seed") == expected["seed"],
                "split": summary.get("split_hash") == dev_split_hash,
                "manifest": summary.get("manifest_sha256") == manifest_sha256,
                "dev_file": summary.get("split_file_sha256") == dev_sha256,
                "max_new_tokens": summary.get("max_new_tokens")
                == expected["eval_max_new_tokens"],
                "harness_mode": summary.get("harness_mode")
                == expected["eval_harness_mode"],
                "generation": summary.get("generation")
                == expected["eval_generation"],
                "protocol_lock": summary.get("experiment_protocol_lock_sha256")
                == protocol_lock_sha256,
                "checkpoint_path": summary.get("checkpoint")
                == str(expected_checkpoint),
                "checkpoint_hash": summary.get("checkpoint_sha256")
                == directory_sha256(expected_checkpoint),
            }
            if not all(checks.values()):
                raise ValueError(
                    f"checkpoint evaluation binding mismatch at {path}: {checks}"
                )
        rows.append(
            {
                "checkpoint": checkpoint_name,
                "s_on": summary["s_on"],
                "s_off": summary["s_off"],
                "dependence_gap": summary["dependence_gap"],
                "first_pass": summary["first_pass"],
                "first_attempt_accuracy": summary["first_attempt_accuracy"],
                "intervention_rate": summary["intervention_rate"],
                "mean_preference_margin_b_over_a": summary.get(
                    "mean_preference_margin_b_over_a"
                ),
            }
        )
    return rows


def analyze(protocol: dict, base: dict, arms: dict[str, dict[str, Any]]) -> dict:
    naive = arms["naive_h_grpo"]
    mask = arms["mask_correction"]
    rescue = arms["rescuecredit"]
    naive_eval = naive["eval"]
    rescue_eval = rescue["eval"]
    rescued_signal = naive["training_signal"]["rescued"]
    mask_rescued = mask["training_signal"]["rescued"]
    rescue_rescued = rescue["training_signal"]["rescued"]
    rescue_audits = rescue["run"]["audit_stats"]

    mechanism_checks = {
        "enough_counterfactual_rescue_events": rescued_signal["events"]
        >= protocol["gate"]["min_diagnostic_rescue_events"],
        "naive_never_receives_negative_prefix_signal_on_rescues": (
            rescued_signal["negative_prefix_rate"] is not None
            and rescued_signal["negative_prefix_rate"]
            <= protocol["gate"]["max_naive_negative_prefix_rate_on_rescues"]
        ),
        "naive_harness_dependence_is_positive": naive_eval["dependence_gap"] > 0,
        "harness_actually_rescues_tasks": naive_eval.get("rescued_tasks", 0) > 0,
        "mask_observes_same_rescue_support": mask_rescued["events"]
        >= protocol["gate"]["min_diagnostic_rescue_events"],
        "mask_zeros_prefix_credit_on_rescues": (
            mask_rescued["zero_prefix_rate"] is not None
            and abs(mask_rescued["zero_prefix_rate"] - 1.0) <= 1e-12
        ),
        "mask_uses_correction_loss": mask["training_signal"][
            "nonzero_correction_updates"
        ]
        > 0,
        "rescue_observes_same_rescue_support": rescue_rescued["events"]
        >= protocol["gate"]["min_diagnostic_rescue_events"],
        "rescue_has_valid_full_shadow_audits": (
            rescue_audits["valid_audits"]
            >= protocol["gate"]["min_diagnostic_rescue_events"]
            and rescue["run"]["shadow_steps"] > 0
        ),
        "rescue_g0_matches_diagnostic_shadow": (
            rescue["training_signal"]["shadow_alignment_checks"]
            >= protocol["gate"]["min_diagnostic_rescue_events"]
            and rescue["training_signal"]["shadow_alignment_rate"] == 1.0
        ),
        "rescue_uses_correction_loss": rescue["training_signal"][
            "nonzero_correction_updates"
        ]
        > 0,
        "rescue_restores_negative_prefix_signal": (
            rescue_rescued["negative_prefix_rate"] is not None
            and rescued_signal["negative_prefix_rate"] is not None
            and rescue_rescued["negative_prefix_rate"]
            > rescued_signal["negative_prefix_rate"]
        ),
    }
    method_checks = {
        "rescue_improves_s_off_vs_naive": rescue_eval["s_off"]
        > naive_eval["s_off"],
        "rescue_improves_first_attempt_vs_naive": rescue_eval[
            "first_attempt_accuracy"
        ]
        > naive_eval["first_attempt_accuracy"],
        "rescue_reduces_intervention_rate_vs_naive": rescue_eval[
            "intervention_rate"
        ]
        < naive_eval["intervention_rate"],
        "rescue_not_worse_than_mask_on_s_off": rescue_eval["s_off"]
        >= mask["eval"]["s_off"],
    }
    return {
        "status": "completed",
        "stage": "harness_credit_blindness_seed42",
        "scope": protocol["scope"],
        "mechanism_supported": all(mechanism_checks.values()),
        "method_supported": all(method_checks.values()),
        "passed": all(mechanism_checks.values()) and all(method_checks.values()),
        "mechanism_checks": mechanism_checks,
        "method_checks": method_checks,
        "base": base,
        "methods": arms,
        "primary_deltas": {
            "rescue_minus_naive_s_off": rescue_eval["s_off"]
            - naive_eval["s_off"],
            "rescue_minus_naive_first_attempt_accuracy": rescue_eval[
                "first_attempt_accuracy"
            ]
            - naive_eval["first_attempt_accuracy"],
            "rescue_minus_naive_intervention_rate": rescue_eval[
                "intervention_rate"
            ]
            - naive_eval["intervention_rate"],
        },
        "interpretation": (
            "The controlled Harness hides policy failures from Naive GRPO, and "
            "RescueCredit improves autonomous behavior."
            if all(mechanism_checks.values()) and all(method_checks.values())
            else "Keep mechanism and method verdicts separate; do not claim a "
            "RescueCredit performance advantage unless method_supported is true."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = read_json(args.protocol_lock)
    if protocol.get("status") != "frozen_before_training":
        raise ValueError("protocol was not frozen before training")
    if directory_sha256(Path(protocol["base_model"]["path"])) != protocol[
        "base_model"
    ]["directory_sha256"]:
        raise ValueError("base model changed after protocol freeze")
    for source, expected_hash in protocol.get("source_sha256", {}).items():
        if file_sha256(Path(source)) != expected_hash:
            raise ValueError(f"source changed after protocol freeze: {source}")
    protocol_lock_sha256 = file_sha256(args.protocol_lock)

    base = read_json(args.base_eval)
    verify_evaluation_artifacts(args.base_eval.parent, base)
    if base.get("checkpoint_sha256") != protocol["base_model"][
        "directory_sha256"
    ]:
        raise ValueError("base evaluation is not bound to the frozen base model")
    expected = protocol["config"]
    base_eval_checks = {
        "seed": base.get("seed") == expected["seed"],
        "max_new_tokens": base.get("max_new_tokens")
        == expected["eval_max_new_tokens"],
        "harness_mode": base.get("harness_mode")
        == expected["eval_harness_mode"],
        "generation": base.get("generation") == expected["eval_generation"],
        "split_hash": base.get("split_hash") == protocol["data"]["dev_split_hash"],
        "manifest": base.get("manifest_sha256")
        == protocol["data"]["manifest_sha256"],
        "dev_file": base.get("split_file_sha256")
        == protocol["data"]["dev_sha256"],
        "protocol_lock": base.get("experiment_protocol_lock_sha256")
        == protocol_lock_sha256,
    }
    if not all(base_eval_checks.values()):
        raise ValueError(f"base evaluation config mismatch: {base_eval_checks}")
    arms: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        root = args.root / method
        run = read_json(root / "run_summary.json")
        final_eval_path = root / "eval_final" / "eval_summary.json"
        final_eval = read_json(final_eval_path)
        verify_evaluation_artifacts(final_eval_path.parent, final_eval)
        if run["method"] != method or final_eval["method"] != method:
            raise ValueError(f"method identity mismatch for {method}")
        if run["split_hash"] != protocol["data"]["train_split_hash"]:
            raise ValueError(f"train split mismatch for {method}")
        if final_eval["split_hash"] != protocol["data"]["dev_split_hash"]:
            raise ValueError(f"dev split mismatch for {method}")
        if run.get("experiment_protocol_lock_sha256") != protocol_lock_sha256:
            raise ValueError(f"protocol lock binding mismatch for {method}")
        if final_eval.get("checkpoint") != run.get("checkpoint"):
            raise ValueError(f"final evaluation checkpoint path mismatch for {method}")
        if final_eval.get("checkpoint_sha256") != run.get("checkpoint_sha256"):
            raise ValueError(f"final evaluation checkpoint hash mismatch for {method}")
        if directory_sha256(Path(run["checkpoint"])) != run.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"checkpoint changed after training for {method}")
        comparable = run["comparability"]
        checks = {
            "seed": run["seed"] == expected["seed"],
            "main_budget": run["budget_mode"] == "main"
            and run["main_interaction_budget"]
            == expected["main_interaction_budget"],
            "exact_main_steps": run["main_steps"]
            == expected["main_interaction_budget"]
            and run["budget_unused"] == 0
            and run["budget_overshoot"] == 0,
            "authoritative_unique_cost": run.get(
                "authoritative_unique_interaction_steps"
            )
            == (
                run["main_steps"]
                + run["shadow_steps"]
                + run["failed_replay_steps"]
                + run["diagnostic_full_shadow"]["unique_extra_steps"]
            ),
            "world_size": run["world_size"] == expected["world_size"],
            "group_size": comparable["group_size"] == expected["group_size"],
            "max_new_tokens": comparable["max_new_tokens"]
            == expected["max_new_tokens"],
            "max_shadow_steps": comparable["max_shadow_steps"]
            == expected["max_shadow_steps"],
            "policy_epochs": comparable["policy_epochs"]
            == expected["policy_epochs"],
            "learning_rate": comparable["learning_rate"]
            == expected["learning_rate"],
            "use_lora": comparable["use_lora"] is True,
            "fp32": comparable["fp32"] is True,
            "diagnostic_enabled": run["diagnostic_full_shadow"]["enabled"]
            is True,
            "force_shadow_credit_contract": comparable.get(
                "force_shadow_credit", False
            )
            is (method == "rescuecredit"),
            "strict_main_budget": comparable.get("strict_main_budget") is True,
            "lambda_corr": comparable.get("lambda_corr")
            == expected["lambda_corr"],
            "eval_seed": final_eval.get("seed") == expected["seed"],
            "eval_max_new_tokens": final_eval.get("max_new_tokens")
            == expected["eval_max_new_tokens"],
            "eval_harness_mode": final_eval.get("harness_mode")
            == expected["eval_harness_mode"],
            "eval_generation": final_eval.get("generation")
            == expected["eval_generation"],
            "eval_manifest": final_eval.get("manifest_sha256")
            == protocol["data"]["manifest_sha256"],
            "eval_dev_file": final_eval.get("split_file_sha256")
            == protocol["data"]["dev_sha256"],
            "eval_protocol_lock": final_eval.get(
                "experiment_protocol_lock_sha256"
            )
            == protocol_lock_sha256,
            "audit_probability": comparable.get("audit_probability")
            == expected["audit_probability"],
            "audit_warm_start": comparable.get("audit_warm_start_events")
            == expected["audit_warm_start_events"],
            "save_every": comparable.get("save_every") == expected["save_every"],
            "max_updates": comparable.get("max_updates")
            == expected["max_updates"],
            "temperature": comparable.get("temperature")
            == expected["temperature"],
            "clip_eps": comparable.get("clip_eps") == expected["clip_eps"],
            "kl_coef": comparable.get("kl_coef") == expected["kl_coef"],
            "lambda_causal": comparable.get("lambda_causal")
            == expected["lambda_causal"],
            "visible_curriculum_fraction": comparable.get(
                "visible_curriculum_fraction"
            )
            == expected["visible_curriculum_fraction"],
            "total_interaction_budget": comparable.get("interaction_budget")
            == expected["total_interaction_budget"],
            "model_revision": run.get("model_revision")
            == expected["model_revision"],
        }
        if not all(checks.values()):
            raise ValueError(f"comparability failure for {method}: {checks}")
        declared_logs = run.get("training_logs", [])
        actual_logs = sorted(root.glob("train_rank*.jsonl"))
        if {entry.get("path") for entry in declared_logs} != {
            path.name for path in actual_logs
        }:
            raise ValueError(f"training log set mismatch for {method}")
        for entry in declared_logs:
            path = root / entry["path"]
            row_count = sum(
                1
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if file_sha256(path) != entry.get("sha256") or row_count != entry.get(
                "rows"
            ):
                raise ValueError(f"training log artifact mismatch: {path}")
        signal = training_signal(
            root,
            expected_method=method,
            expected_config_hash=run["config_hash"],
            expected_run_id=run["run_id"],
            expected_lambda_corr=expected["lambda_corr"],
        )
        if signal["train_rows"] <= 0 or signal["realized_group_sizes"] != [
            expected["group_size"]
        ]:
            raise ValueError(
                f"realized GRPO group cardinality mismatch for {method}: {signal}"
            )
        if signal["train_rows"] != run["sampling"]["trained_updates"]:
            raise ValueError(f"training log/update count mismatch for {method}")
        arms[method] = {
            "run": run,
            "eval": final_eval,
            "training_signal": signal,
            "curve": checkpoint_curve(
                root,
                method=method,
                expected=expected,
                dev_split_hash=protocol["data"]["dev_split_hash"],
                manifest_sha256=protocol["data"]["manifest_sha256"],
                dev_sha256=protocol["data"]["dev_sha256"],
                protocol_lock_sha256=protocol_lock_sha256,
            ),
            "artifact_hashes": {
                "run_summary": file_sha256(root / "run_summary.json"),
                "final_eval": file_sha256(final_eval_path),
            },
        }

    result = analyze(protocol, base, arms)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
