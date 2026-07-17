from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_evaluator_hardens_protocol_and_official_score_contract() -> None:
    source = (ROOT / "scripts/evaluate_route_a_bounded.py").read_text(encoding="utf-8")
    assert "this preregistered diagnostic requires seed=42 and horizons=4 8" in source
    assert "seed + task_index" in source
    assert "official_report_score_missing" in source
    assert "score = official_score" not in source
    assert "protocol_lock_validated" in source


def test_worker_errors_are_not_cached_and_prefixes_are_verified() -> None:
    source = (ROOT / "scripts/evaluate_route_a_bounded.py").read_text(encoding="utf-8")
    error_branch = source.split(
        "# API/protocol failures are never cached and invalidate the branch.", 1
    )[1].split("self.cache[key] = value", 1)[0]
    assert 'if response["status"] == "error"' in error_branch
    assert "return response, key" in error_branch
    assert "trace_prefix_matches" in source
    assert "horizon_prefix_match_a" in source


def test_runner_freezes_inputs_before_sanity_or_full_evaluation() -> None:
    source = (
        ROOT / "scripts/cloud/run_route_a_appworld_bounded.sh"
    ).read_text(encoding="utf-8")
    freeze = source.index("scripts/freeze_route_a_bounded_protocol.py")
    evaluate = source.index("scripts/evaluate_route_a_bounded.py")
    assert freeze < evaluate
    assert source.count('--protocol-lock "$LOCK"') == 2
    assert "--limit 3" in source
