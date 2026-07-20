import copy

import pytest

from rescuecredit.paired_task_statistics import align_oof_rows, paired_task_analysis


def _rows(scores, routed=None):
    labels = [0, 1, 0, 1, 0, 1]
    tasks = ["a", "a", "b", "b", "c", "c"]
    routed = routed or [False] * len(labels)
    return [
        {
            "event_id": str(index),
            "task_id_hash": task,
            "label": label,
            "active_raw_score": score,
            "probed": True,
            "routed_to_a": route,
        }
        for index, (task, label, score, route) in enumerate(
            zip(tasks, labels, scores, routed, strict=True)
        )
    ]


def test_alignment_rejects_label_or_task_drift():
    left = _rows([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    right = copy.deepcopy(left)
    right[0]["task_id_hash"] = "changed"
    with pytest.raises(ValueError, match="task identities"):
        align_oof_rows(left, right)


def test_alignment_rejects_routing_without_probe():
    left = _rows([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    right = copy.deepcopy(left)
    right[0]["routed_to_a"] = True
    right[0]["probed"] = False
    with pytest.raises(ValueError, match="unprobed"):
        align_oof_rows(left, right)


def test_paired_task_analysis_is_deterministic_and_detects_direction():
    v7 = _rows([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    v9 = _rows([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    first = paired_task_analysis(
        v7,
        v9,
        bootstrap_replicates=200,
        permutation_replicates=200,
        seed=42,
        alpha=0.05,
    )
    second = paired_task_analysis(
        v7,
        v9,
        bootstrap_replicates=200,
        permutation_replicates=200,
        seed=42,
        alpha=0.05,
    )
    assert first == second
    assert first["observed"]["v9_minus_v7"]["roc_auc"] < 0
    assert first["positive_routing_claim_supported"] is False
    assert first["secondary_metric_p_values_exploratory"] is True


def test_analysis_reports_operating_point_deltas():
    v7 = _rows(
        [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
        [False, True, False, True, False, True],
    )
    v9 = _rows(
        [0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
        [True, True, False, True, False, False],
    )
    result = paired_task_analysis(
        v7,
        v9,
        bootstrap_replicates=100,
        permutation_replicates=100,
        seed=7,
        alpha=0.05,
    )
    delta = result["observed"]["v9_minus_v7"]
    assert set(delta) == {"roc_auc", "reverse_recall", "rescue_drop", "probe_rate"}
