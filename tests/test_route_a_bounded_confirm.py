from scripts.analyze_route_a_bounded_confirm import cluster_bootstrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cluster_bootstrap_strictly_positive_support() -> None:
    event_deltas = {f"event-{index}": [0.1] for index in range(20)}
    result = cluster_bootstrap(event_deltas)
    assert result["clusters"] == 20
    assert result["mean"] == 0.1
    assert result["ci95_lower"] > 0.0


def test_cluster_bootstrap_rejects_one_sparse_win() -> None:
    event_deltas = {f"event-{index}": [0.0] for index in range(55)}
    event_deltas["event-0"] = [0.1]
    result = cluster_bootstrap(event_deltas)
    assert result["mean"] > 0.0
    assert result["ci95_lower"] == 0.0


def test_confirmatory_worker_has_explicit_isolation_boundary() -> None:
    source = (ROOT / "scripts/evaluate_route_a_bounded.py").read_text(
        encoding="utf-8"
    )
    assert "WORKER_ENV_ALLOWLIST" in source
    assert "env=worker_env" in source
    assert "cwd=sandbox_dir" in source
    assert 'prefix="rescuecredit_continuation_"' in source
    assert '"APPWORLD_ROOT"' not in source.split(
        "WORKER_ENV_ALLOWLIST = frozenset(", 1
    )[1].split(")", 1)[0]


def test_confirmatory_analyzer_recomputes_primary_statistics() -> None:
    source = (ROOT / "scripts/analyze_route_a_bounded_confirm.py").read_text(
        encoding="utf-8"
    )
    assert "primary = summarize_horizon(rows, 8)" in source
    assert '"primary_recomputed_from_rows"' in source
    assert '"same_valid_event_subset_across_seeds"' in source
