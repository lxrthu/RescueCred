from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repair_selection_is_outcome_independent() -> None:
    source = (ROOT / "scripts/repair_route_a_v21_balanced_data.py").read_text(
        encoding="utf-8"
    )
    selection = source.split("# The cap is deliberately", 1)[1].split(
        "bank_dir =", 1
    )[0]
    assert 'row["delta"]' not in selection
    assert 'row["decision"]' not in selection
    assert "sorted(by_task[task_id]" in selection
    assert ": args.max_events_per_task" in selection


def test_repair_runner_freezes_round_task_cap() -> None:
    source = (ROOT / "scripts/cloud/run_route_a_v21c_repair.sh").read_text(
        encoding="utf-8"
    )
    assert "--max-events-per-task 10" in source
    assert "--max-task-nonzero-share 0.10" in source
    assert "--min-replay-valid-rate 0.90" in source


def test_repaired_data_is_bound_to_v3_label() -> None:
    prepare = (ROOT / "scripts/prepare_route_a_v21_data.py").read_text(
        encoding="utf-8"
    )
    gate = (ROOT / "scripts/check_route_a_v21_data.py").read_text(
        encoding="utf-8"
    )
    assert '"rescuecredit_v3"' in prepare
    assert "Mask vs V3" in gate
