import math

from rescuecredit.rapg import (
    cross_fitted_residual_predictions,
    public_hash_features,
    simulate_fixed_propensity_audits,
    solve_expected_budget,
    stable_seed,
)


def test_expected_budget_solver_respects_floor_cost_and_order():
    probabilities = solve_expected_budget(
        [1.0, 2.0, 4.0],
        expected_budget=1.2,
        p_min=0.1,
        costs=[1.0, 1.0, 1.0],
    )
    assert math.isclose(sum(probabilities), 1.2, abs_tol=1e-6)
    assert probabilities[0] >= 0.1
    assert probabilities[0] <= probabilities[1] <= probabilities[2]


def test_public_hash_features_are_deterministic_and_explicit():
    first = public_hash_features([{"tool": "send"}, "visible"], dimension=32)
    second = public_hash_features([{"tool": "send"}, "visible"], dimension=32)
    private = public_hash_features(
        [{"tool": "send"}, "visible", {"official_score": 1.0}], dimension=32
    )
    assert first == second
    assert first != private
    assert stable_seed(42, "event") == stable_seed(42, "event")


def test_task_crossfit_outputs_finite_predictions_without_overlap():
    import pytest

    torch = pytest.importorskip("torch")

    features = torch.tensor(
        [[float(i), float(i % 3)] for i in range(30)], dtype=torch.float32
    )
    outcomes = 0.2 * features[:, 0] - 0.1 * features[:, 1]
    groups = [f"task-{i // 3}" for i in range(30)]
    means, scales, fold_ids, audits = cross_fitted_residual_predictions(
        features,
        outcomes,
        groups,
        folds=5,
        seed=42,
        ridge_alpha=1.0,
    )
    assert torch.isfinite(means).all()
    assert torch.isfinite(scales).all()
    assert (scales > 0).all()
    assert min(fold_ids) == 0 and max(fold_ids) == 4
    assert all(row["task_overlap"] == 0 for row in audits)


def test_rapg_audit_estimator_is_unbiased_and_matched_cost():
    import pytest

    torch = pytest.importorskip("torch")

    generator = torch.Generator().manual_seed(7)
    events = 100
    score_sketches = torch.randn(events, 4, generator=generator)
    score_norms = score_sketches.norm(dim=1).clamp_min(0.1)
    outcomes = torch.linspace(-1.0, 1.0, events).square()
    means = torch.zeros(events)
    scales = outcomes.abs().clamp_min(0.01)
    summary, artifact = simulate_fixed_propensity_audits(
        score_sketches=score_sketches,
        score_norms=score_norms,
        outcomes=outcomes,
        executed_returns=torch.zeros(events),
        replaced=torch.ones(events, dtype=torch.bool),
        mean_predictions=means,
        scale_predictions=scales,
        groups=[f"task-{index // 5}" for index in range(events)],
        audit_rate=0.2,
        p_min=0.02,
        replicates=5000,
        seed=99,
    )
    assert summary["methods"]["rapg"]["projected_bias_relative"] < 0.1
    assert math.isclose(
        summary["methods"]["uniform"]["expected_audits"],
        summary["methods"]["rapg"]["expected_audits"],
        abs_tol=5e-6,
    )
    assert summary["methods"]["rapg"]["full_gradient_design_mse"] < summary[
        "methods"
    ]["uniform"]["full_gradient_design_mse"]
    assert artifact["methods"]["rapg"]["gradient_estimates"].shape == (5000, 4)
