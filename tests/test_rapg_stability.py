from rescuecredit.rapg_stability import summarize_task_stability


def _rows(rapg_values):
    return [
        {
            "task_id": f"task-{index}",
            "events": 5,
            "uniform_numerator": 10.0,
            "rapg_numerator": rapg,
        }
        for index, rapg in enumerate(rapg_values)
    ]


def test_task_stability_recognizes_robust_gain():
    summary = summarize_task_stability(
        _rows([5.0] * 8), bootstrap_replicates=500, seed=7
    )

    assert summary["classification"] == (
        "surrogate_signal_robust_but_frozen_gate_failed"
    )
    assert summary["observed_mse_gain_over_uniform"] == 0.5
    assert summary["leave_one_task_out"]["minimum"] == 0.5
    assert summary["task_cluster_bootstrap"]["lower"] == 0.5
    assert summary["positive_improvement_concentration"]["top1_share"] == 0.125


def test_task_stability_detects_single_task_concentration():
    summary = summarize_task_stability(
        _rows([0.0, 10.0, 10.0, 10.0, 10.0]),
        bootstrap_replicates=1_000,
        seed=11,
    )

    assert summary["classification"] == "surrogate_gain_task_concentrated"
    assert summary["observed_mse_gain_over_uniform"] == 0.2
    assert summary["leave_one_task_out"]["minimum"] == 0.0
    assert summary["positive_improvement_concentration"]["top1_share"] == 1.0
    assert summary["task_cluster_bootstrap"]["lower"] == 0.0


def test_task_bootstrap_is_seed_deterministic():
    rows = _rows([2.0, 4.0, 6.0, 8.0, 10.0])
    first = summarize_task_stability(rows, bootstrap_replicates=500, seed=123)
    second = summarize_task_stability(rows, bootstrap_replicates=500, seed=123)

    assert first["task_cluster_bootstrap"] == second["task_cluster_bootstrap"]
