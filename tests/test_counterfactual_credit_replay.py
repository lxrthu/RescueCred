from __future__ import annotations

from pathlib import Path

import pytest


def test_counterfactual_credit_replay_recovers_controlled_oracle():
    torch = pytest.importorskip("torch")
    from rescuecredit.counterfactual_credit_replay import (
        replay_counterfactual_credit,
    )

    scores = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    proposal = torch.tensor([0.0, 1.0, 0.0, 1.0])
    executed = torch.tensor([1.0, 0.0, 1.0, 0.0])
    summary, artifact = replay_counterfactual_credit(
        score_sketches=scores,
        proposal_returns=proposal,
        executed_returns=executed,
        replaced=torch.ones(4, dtype=torch.bool),
        mean_predictions=proposal.clone(),
        groups=["task-a", "task-a", "task-b", "task-b"],
        propensities=(0.2, 0.3),
        primary_propensity=0.3,
        replicates=500,
        seed=42,
    )
    assert summary["firewall_oracle_distance_ratio_vs_naive"] == pytest.approx(0.5)
    assert summary["task_results"][
        "firewall_task_improvement_fraction_vs_naive"
    ] == pytest.approx(1.0)
    assert summary["randomized_shadow"]["p30"]["aipw"][
        "projected_gradient_mse"
    ] == pytest.approx(0.0)
    assert summary["diagnostics"]["naive_rescue_like_credit_error_energy"] > 0
    assert summary["analytic_aipw_identity"] is True
    assert artifact["naive_gradient"].tolist() != artifact[
        "oracle_gradient"
    ].tolist()


def test_counterfactual_credit_replay_rejects_invalid_propensity():
    torch = pytest.importorskip("torch")
    from rescuecredit.counterfactual_credit_replay import (
        replay_counterfactual_credit,
    )

    with pytest.raises(ValueError, match="invalid fixed shadow propensities"):
        replay_counterfactual_credit(
            score_sketches=torch.ones(2, 1),
            proposal_returns=torch.ones(2),
            executed_returns=torch.ones(2),
            replaced=torch.ones(2, dtype=torch.bool),
            mean_predictions=torch.zeros(2),
            groups=["a", "b"],
            propensities=(0.2,),
            primary_propensity=0.3,
            replicates=100,
        )


def test_task_bootstrap_gain_is_deterministic_and_directional():
    from rescuecredit.counterfactual_credit_replay import task_bootstrap_gain

    rows = [
        {
            "naive_oracle_squared_distance": 2.0,
            "method_error": 1.0,
        },
        {
            "naive_oracle_squared_distance": 4.0,
            "method_error": 1.0,
        },
    ]
    first = task_bootstrap_gain(rows, "method_error", replicates=500, seed=7)
    second = task_bootstrap_gain(rows, "method_error", replicates=500, seed=7)
    assert first == second
    assert first["lower95"] > 0.0


def test_counterfactual_credit_runner_is_cpu_only_and_fail_closed():
    runner = Path(
        "scripts/cloud/run_toolsandbox_counterfactual_credit_replay_seed42.sh"
    ).read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" not in runner
    assert "feasibility_gate.json" in runner
    assert "behavior_ledger.jsonl" in runner
    assert 'exit "$STATUS"' in runner


def test_counterfactual_credit_plan_marks_historical_scope():
    plan = Path(
        "refine-logs/COUNTERFACTUAL_CREDIT_REPLAY_PLAN_20260721_204520.md"
    ).read_text(encoding="utf-8")
    assert "历史 RAPG development surrogate" in plan
    assert "不能作为 untouched confirmation" in plan
