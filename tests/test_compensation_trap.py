from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from rescuecredit.compensation_trap import (
    build_collision_audit,
    deterministic_split,
    exact_signature_payload,
    private_projection,
    public_projection,
    validate_benchmark_package_data,
)
from rescuecredit.toolsandbox_credit import (
    OFFICIAL_SCORE_SOURCE,
    lexicographic_counterfactual_regret,
)
from scripts.build_compensation_trap_benchmark import validated_official_credit


def _raw(event: str, task: str, decision: str, argument: str) -> dict:
    return {
        "event_id": event,
        "task_id_hash": task,
        "scenario_name": "send_message_cellular_off_1_distraction_tools",
        "reference_free_prefix_steps": 1,
        "action_a": {"tool": "send_message", "arguments": {"content": argument}},
        "action_b": {"tool": "send_message", "arguments": {"content": argument + "!"}},
        "treatment_visible_history": [{"sender": "USER", "content": "send it"}],
        "treatment_public_tool_schemas": [{"name": "send_message"}],
        "decision": decision,
        "decision_basis": "similarity",
        "decision_value": 1.0,
        "causal_weight": 1.0,
    }


def test_exact_collision_is_public_only_and_cross_task():
    rescue = _raw("a", "task-a", "rescue_preference", "hello")
    reverse = _raw("b", "task-b", "reverse_preference", "world")
    public = [public_projection(rescue, "s"), public_projection(reverse, "s")]
    private = [private_projection(rescue, "s"), private_projection(reverse, "s")]
    assert exact_signature_payload(public[0]) == exact_signature_payload(public[1])
    summary, pairs = build_collision_audit(public, private, similarity_threshold=0.1)
    assert summary["exact_cross_task_opposing_pairs"] == 1
    assert summary["empirical_exact_signature_conditional_bayes_error"] == pytest.approx(0.5)
    assert any(row["kind"] == "exact_signature" for row in pairs)


def test_collision_rejects_duplicate_and_misaligned_events():
    raw = _raw("a", "task-a", "rescue_preference", "hello")
    public = public_projection(raw, "s")
    private = private_projection(raw, "s")
    with pytest.raises(ValueError, match="duplicate"):
        build_collision_audit([public, public], [private, private])
    other = dict(private)
    other["event_id"] = "other"
    with pytest.raises(ValueError, match="sets differ"):
        build_collision_audit([public], [other])


def test_approximate_nearest_neighbors_are_selected_without_labels():
    raws = [
        _raw("a", "task-a", "rescue_preference", "same public content"),
        _raw("b", "task-b", "rescue_preference", "same public content"),
        _raw("c", "task-c", "reverse_preference", "entirely unrelated value xyz"),
    ]
    public = [public_projection(raw, "s") for raw in raws]
    private = [private_projection(raw, "s") for raw in raws]
    summary, _ = build_collision_audit(public, private, similarity_threshold=0.1)
    assert summary["approximate_cross_task_mutual_nearest_pairs"] == 0


def test_split_is_deterministic_and_task_level():
    assert deterministic_split("task-a") == deterministic_split("task-a")
    assert deterministic_split("task-a") in {"train", "development", "test"}


def test_runner_is_fail_closed_and_cpu_only():
    runner = Path("scripts/cloud/run_compensation_trap_cpu_seed42.sh").read_text(
        encoding="utf-8"
    )
    assert "CUDA_VISIBLE_DEVICES" not in runner
    assert 'exit "$STATUS"' in runner
    assert "similarity-threshold 0.90" in runner


def test_public_projection_has_no_outcome_fields():
    raw = _raw("a", "task-a", "rescue_preference", "hello")
    public = public_projection(raw, "s")
    serialized = json.dumps(public, sort_keys=True)
    assert "decision" not in serialized
    assert "branch_a" not in serialized
    assert "branch_b" not in serialized


def _official_branch(similarity: float) -> dict:
    return {
        "valid": True,
        "score": {
            "source": OFFICIAL_SCORE_SOURCE,
            "similarity": similarity,
            "turn_count": 1,
        },
        "score_trace": [
            {"source": OFFICIAL_SCORE_SOURCE, "similarity": similarity}
        ],
        "padded_similarity_trace": [similarity],
        "progress_auc": similarity,
        "tool_errors": 0,
        "steps": 1,
    }


def test_stored_credit_must_match_official_recomputation():
    branch_a = _official_branch(0.0)
    branch_b = _official_branch(1.0)
    credit = lexicographic_counterfactual_regret(branch_a, branch_b, horizon=1)
    raw = {
        "event_id": "event",
        "credit_mode": "lexicographic_v4",
        "branch_a": branch_a,
        "branch_b": branch_b,
        "decision": credit["decision"],
        "decision_basis": credit["decision_basis"],
        "decision_value": credit["decision_value"],
        "causal_weight": credit["causal_weight"],
        "credit_components": credit["components"],
    }
    assert validated_official_credit(raw, horizon=1, atol=1e-12) == credit
    raw["decision"] = "zero_delta"
    with pytest.raises(ValueError, match="official recomputation"):
        validated_official_credit(raw, horizon=1, atol=1e-12)


def test_shared_package_validation_rejects_label_split_and_count_tampering():
    raws = [
        _raw("a", "unused", "rescue_preference", "hello"),
        _raw("b", "unused", "reverse_preference", "world"),
    ]
    for index, raw in enumerate(raws):
        raw["scenario_name"] = f"scenario_{index}"
        raw["task_id_hash"] = hashlib.sha256(
            raw["scenario_name"].encode("utf-8")
        ).hexdigest()
    public = [public_projection(raw, "source") for raw in raws]
    private = [private_projection(raw, "source") for raw in raws]
    splits = [
        {
            "event_id": row["event_id"],
            "task_id_hash": row["task_id_hash"],
            "split": deterministic_split(row["task_id_hash"]),
        }
        for row in public
    ]
    schema = {
        "version": "compensation_trap_benchmark_v1",
        "public_fields": list(public[0]),
        "private_fields": list(private[0]),
        "labels": ["rescue_preference", "reverse_preference"],
        "split_rule": "sha256(task_id_hash) prefix modulo 10: 0-5 train, 6-7 development, 8-9 test",
    }
    split_event_counts = {}
    split_task_counts = {}
    for split in ("train", "development", "test"):
        split_event_counts[split] = sum(row["split"] == split for row in splits)
        split_task_counts[split] = len(
            {row["task_id_hash"] for row in splits if row["split"] == split}
        )
    split_event_counts = {
        key: value for key, value in split_event_counts.items() if value
    }
    manifest = {
        "status": "completed",
        "version": "compensation_trap_benchmark_v1",
        "official_branch_evidence_recomputed": True,
        "credit_mode": "lexicographic_v4",
        "horizon": 8,
        "atol": 1e-12,
        "events": 2,
        "tasks": 2,
        "label_counts": {"rescue_preference": 1, "reverse_preference": 1},
        "split_event_counts": split_event_counts,
        "split_task_counts": split_task_counts,
        "sources": [{"name": "source"}],
    }
    assert all(
        validate_benchmark_package_data(public, private, splits, schema, manifest).values()
    )
    bad_private = [dict(row) for row in private]
    bad_private[0]["decision"] = "third_label"
    assert not validate_benchmark_package_data(
        public, bad_private, splits, schema, manifest
    )["label_boundary"]
    bad_splits = [dict(row) for row in splits]
    bad_splits[0]["task_id_hash"] = bad_splits[1]["task_id_hash"]
    assert not validate_benchmark_package_data(
        public, private, bad_splits, schema, manifest
    )["public_private_task_alignment"]
    bad_manifest = dict(manifest)
    bad_manifest["split_event_counts"] = {}
    assert not validate_benchmark_package_data(
        public, private, splits, schema, bad_manifest
    )["statistics_recomputed"]
