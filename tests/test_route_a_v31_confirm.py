from pathlib import Path

from scripts.analyze_route_a_v31_confirm import cluster_bootstrap, equivalent
from scripts.freeze_route_a_v31_confirm_protocol import (
    AGGREGATE_THRESHOLDS,
    CONFIRMATORY_SEEDS,
    expected_config,
    expected_mask_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_confirmatory_config_changes_only_seed() -> None:
    configs = [expected_config(seed) for seed in CONFIRMATORY_SEEDS]
    assert [config["seed"] for config in configs] == [43, 44, 45]
    stripped = [{k: v for k, v in config.items() if k != "seed"} for config in configs]
    assert stripped[0] == stripped[1] == stripped[2]
    mask = expected_mask_config(43)
    assert mask["method"] == "mask"
    assert mask["absolute_margin_coef"] == 0.0


def test_aggregate_gate_is_replication_not_single_seed() -> None:
    assert AGGREGATE_THRESHOLDS["minimum_positive_score_seeds"] == 3
    assert AGGREGATE_THRESHOLDS["minimum_total_nonzero_events"] == 15
    assert AGGREGATE_THRESHOLDS["require_aggregate_wins_over_losses"] is True


def test_roundoff_equivalence_is_narrow() -> None:
    assert equivalent({"score": 0.1 + 0.2}, {"score": 0.3})
    assert not equivalent({"score": 0.3001}, {"score": 0.3})


def test_bootstrap_is_reported_but_not_a_gate() -> None:
    result = cluster_bootstrap({f"event-{index}": [0.1] for index in range(10)})
    assert result["mean"] == 0.1
    assert result["ci95_lower"] > 0
    assert result["gating_role"] == "reported_not_gated"


def test_evaluator_has_explicit_both_valid_confirmatory_mode() -> None:
    source = (ROOT / "scripts/evaluate_route_a_bounded.py").read_text(
        encoding="utf-8"
    )
    assert '"--development-confirmatory"' in source
    assert '"development_confirmatory": args.development_confirmatory' in source


def test_runner_retrains_both_methods_for_each_seed() -> None:
    source = (
        ROOT / "scripts/cloud/run_route_a_v31_confirm_43_44_45.sh"
    ).read_text(encoding="utf-8")
    assert source.count("train_route_a_preference.py") == 2
    assert "--method mask" in source
    assert "--method v31" in source
    assert "for SEED in 43 44 45" in source
    assert "--development-confirmatory" in source
