import json
import math
import sys
from pathlib import Path

import pytest

from rescuecredit.edit_credit import (
    fold_role,
    select_rescue_constrained_threshold,
    summarize_selection,
)
from rescuecredit.frozen_bank import (
    directory_sha256,
    file_sha256,
    read_jsonl,
    write_jsonl,
)
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash
from scripts.check_toolsandbox_editcredit_gate import main as gate_main
from scripts.freeze_toolsandbox_editcredit_protocol import build_protocol


def _frozen_bank(tmp_path: Path):
    rows = [
        {
            "event_id": f"event-{index}",
            "task_id_hash": f"task-{index % 38}",
            "decision": "rescue_preference" if index < 41 else "reverse_preference",
            "replay_valid": True,
        }
        for index in range(126)
    ]
    train_file = tmp_path / "train.jsonl"
    write_jsonl(train_file, rows)
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "status": "frozen",
            "passed": True,
            "events": 126,
            "train_sha256": file_sha256(train_file),
            "official_branch_metrics_in_training_file": False,
            "protected_outcomes_in_prompt": False,
            "source_event_sha256": "events",
            "source_summary_sha256": "summary",
            "source_protocol_sha256": "protocol",
        },
    )
    data_gate = tmp_path / "data_gate.json"
    write_json(data_gate, {"passed": True, "events": 126})
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    gradient_sanity = tmp_path / "gradient_sanity.json"
    write_json(gradient_sanity, {"passed": True, "checks": {"all": True}})
    protocol = build_protocol(
        train_file, model, manifest, data_gate, gradient_sanity
    )
    lock = tmp_path / "protocol.json"
    write_json(lock, protocol)
    return rows, train_file, manifest, data_gate, gradient_sanity, protocol, lock


def test_gate_rebuilds_labels_splits_and_calibration_from_frozen_bank(
    tmp_path, monkeypatch
):
    (
        rows,
        train_file,
        manifest,
        data_gate,
        gradient_sanity,
        protocol,
        lock,
    ) = _frozen_bank(tmp_path)
    root = tmp_path / "runs"
    assignment = protocol["task_fold_assignment"]
    for fold in range(5):
        for method in ("full_action", "editcredit"):
            directory = root / method / f"fold{fold}"
            eval_dir = directory / "eval"
            adapter = directory / "adapter"
            adapter.mkdir(parents=True)
            eval_dir.mkdir()
            (adapter / "adapter.json").write_text("{}", encoding="utf-8")
            train_rows = [
                row
                for row in rows
                if fold_role(row, assignment=assignment, test_fold=fold, folds=5)
                == "train"
            ]
            run = {
                "status": "completed",
                "method": method,
                "fold": fold,
                "protocol_lock_sha256": file_sha256(lock),
                "adapter": str(adapter),
                "adapter_sha256": directory_sha256(adapter),
                "base_model_sha256": protocol["base_model_sha256"],
                "train_file_sha256": protocol["train_sha256"],
                "train_event_set_hash": event_set_hash(train_rows),
                "train_task_group_ids": sorted(
                    {str(row["task_id_hash"]) for row in train_rows}
                ),
                "source_identity_in_model_input": method != "editcredit",
                "presentation_side_label_auc": 0.5 if method == "editcredit" else None,
                "presentations": protocol["config"]["epochs"]
                * protocol["config"]["presentations_per_epoch"],
                **protocol["config"],
            }
            run_path = directory / "run_summary.json"
            write_json(run_path, run)
            scores = []
            for row in rows:
                role = fold_role(
                    row, assignment=assignment, test_fold=fold, folds=5
                )
                if role not in {"calibration", "test"}:
                    continue
                margin = (
                    1.0
                    if method == "full_action"
                    else 1.0
                    if row["decision"] == "rescue_preference"
                    else -1.0
                )
                scores.append(
                    {
                        "event_id": row["event_id"],
                        "task_id_hash": row["task_id_hash"],
                        "fold": fold,
                        "role": role,
                        "margin_b_over_a": margin,
                        "margin_original_order": margin,
                        "margin_swapped_order": margin,
                        "swap_consistent": True,
                    }
                )
            scores_path = eval_dir / "scores.public.jsonl"
            write_jsonl(scores_path, scores)
            joined = [
                {
                    **score,
                    "decision": next(
                        row["decision"]
                        for row in rows
                        if row["event_id"] == score["event_id"]
                    ),
                }
                for score in scores
            ]
            joined_path = eval_dir / "predictions.joined.jsonl"
            write_jsonl(joined_path, joined)
            threshold = 0.0
            constraint = None
            if method == "editcredit":
                choice = select_rescue_constrained_threshold(
                    [row for row in joined if row["role"] == "calibration"],
                    rescue_delta=0.02,
                )
                threshold = choice.threshold
                constraint = {
                    "rescue_drop": choice.rescue_drop,
                    "reverse_recall": choice.reverse_recall,
                    "route_to_a": choice.route_to_a,
                    "feasible": choice.feasible,
                }
            test_summary = summarize_selection(
                [row for row in joined if row["role"] == "test"],
                threshold=threshold,
            )
            write_json(
                eval_dir / "eval_summary.json",
                {
                    "status": "completed",
                    "method": method,
                    "fold": fold,
                    "protocol_lock_sha256": file_sha256(lock),
                    "run_summary_sha256": file_sha256(run_path),
                    "adapter_sha256": directory_sha256(adapter),
                    "public_scores_sha256": file_sha256(scores_path),
                    "predictions_sha256": file_sha256(joined_path),
                    "selected_threshold": (
                        None if math.isinf(threshold) else threshold
                    ),
                    "calibration_constraint": constraint,
                    "test": {
                        key: value
                        for key, value in test_summary.items()
                        if key != "rows"
                    },
                },
            )
    output = root / "feasibility_gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_toolsandbox_editcredit_gate.py",
            "--protocol-lock",
            str(lock),
            "--root",
            str(root),
            "--train-file",
            str(train_file),
            "--data-manifest",
            str(manifest),
            "--data-gate",
            str(data_gate),
            "--gradient-sanity",
            str(gradient_sanity),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        gate_main()
    assert exit_info.value.code == 0
    gate = json.loads(output.read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert all(gate["integrity_checks"].values())
    assert gate["observed"]["editcredit_constrained"]["balanced_accuracy"] == 1.0

    # A self-consistent outer hash cannot hide a wrong derived mean margin.
    score_path = root / "editcredit" / "fold0" / "eval" / "scores.public.jsonl"
    summary_path = root / "editcredit" / "fold0" / "eval" / "eval_summary.json"
    scores = read_jsonl(score_path)
    scores[0]["margin_b_over_a"] += 0.25
    write_jsonl(score_path, scores)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["public_scores_sha256"] = file_sha256(score_path)
    write_json(summary_path, summary)
    tamper_output = root / "tampered_margin_gate.json"
    tampered_argv = list(sys.argv)
    tampered_argv[-1] = str(tamper_output)
    monkeypatch.setattr(sys, "argv", tampered_argv)
    with pytest.raises(SystemExit) as tamper_exit:
        gate_main()
    assert tamper_exit.value.code == 1
    tamper_gate = json.loads(tamper_output.read_text(encoding="utf-8"))
    assert tamper_gate["integrity_checks"]["score_derivations_recomputed"] is False

    # Restore the score artifact, then prove a wrong base identity is rejected
    # even when the eval summary is rebound to the modified run summary.
    scores[0]["margin_b_over_a"] -= 0.25
    write_jsonl(score_path, scores)
    summary["public_scores_sha256"] = file_sha256(score_path)
    run_path = root / "full_action" / "fold0" / "run_summary.json"
    full_summary_path = root / "full_action" / "fold0" / "eval" / "eval_summary.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["base_model_sha256"] = "wrong-base"
    write_json(run_path, run)
    full_summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
    full_summary["run_summary_sha256"] = file_sha256(run_path)
    write_json(full_summary_path, full_summary)
    write_json(summary_path, summary)
    base_output = root / "tampered_base_gate.json"
    base_argv = list(sys.argv)
    base_argv[-1] = str(base_output)
    monkeypatch.setattr(sys, "argv", base_argv)
    with pytest.raises(SystemExit) as base_exit:
        gate_main()
    assert base_exit.value.code == 1
    base_gate = json.loads(base_output.read_text(encoding="utf-8"))
    assert base_gate["integrity_checks"]["run_and_eval_bound"] is False


def test_gate_rejects_public_score_artifact_with_embedded_label():
    # The full end-to-end fixture above covers the positive path. This focused
    # invariant is enforced before any independent ground-truth join.
    from scripts.check_toolsandbox_editcredit_gate import PROTECTED_SCORE_FIELDS

    assert "decision" in PROTECTED_SCORE_FIELDS
    assert "branch_a" in PROTECTED_SCORE_FIELDS
