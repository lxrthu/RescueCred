from pathlib import Path

from scripts.prepare_route_a_v21_data import counts, task_rank
from scripts.build_appworld_route_a_bank_v21 import _compatible_alternative


ROOT = Path(__file__).resolve().parents[1]


def test_v21_counts_track_task_concentration() -> None:
    rows = [
        {"task_id": "a", "decision": "rescue_preference"},
        {"task_id": "a", "decision": "reverse_preference"},
        {"task_id": "b", "decision": "zero_delta"},
        {"task_id": "c", "decision": "rescue_preference"},
    ]
    result = counts(rows)
    assert result["events"] == 4
    assert result["nonzero_events"] == 3
    assert result["nonzero_tasks"] == 2
    assert result["max_nonzero_events_per_task"] == 2
    assert result["max_task_nonzero_share"] == 2 / 3


def test_v21_task_rank_is_seeded_and_stable() -> None:
    assert task_rank(42, "task") == task_rank(42, "task")
    assert task_rank(42, "task") != task_rank(43, "task")


def test_v21_wrong_value_variants_preserve_scalar_type_family() -> None:
    assert _compatible_alternative("alice", "bob")
    assert _compatible_alternative(1, 2.5)
    assert not _compatible_alternative("1", 1)
    assert not _compatible_alternative(True, 1)


def test_v21_builder_keeps_reference_out_of_public_training_rows() -> None:
    source = (ROOT / "scripts/build_appworld_route_a_bank_v21.py").read_text(
        encoding="utf-8"
    )
    assert "validate_public_record(public)" in source
    assert '"offline_audit.private.jsonl"' in source
    assert '"offline_audit_labels_accessible": False' in source
    assert '"independent_single_field_visible_repairs"' in source
    assert '"wrong_visible_candidate_value"' in source


def test_v21_shadow_worker_is_isolated_and_resumable() -> None:
    source = (ROOT / "scripts/attach_appworld_shadow_credit_v21.py").read_text(
        encoding="utf-8"
    )
    assert "WORKER_ENV_ALLOWLIST" in source
    assert "env=worker_env" in source
    assert "cwd=self.sandbox" in source
    assert "event_checkpoints" in source
    assert '"worker_benchmark_root_in_environment": False' in source
    assert "class WorkerFatalError(RuntimeError):" in source
    assert "import WorkerFatalError" not in source


def test_v21_cloud_gate_is_frozen_before_data_outcomes() -> None:
    source = (
        ROOT / "scripts/cloud/run_route_a_v21_data_expansion.sh"
    ).read_text(encoding="utf-8")
    assert "--min-nonzero-events 100" in source
    assert "--min-nonzero-tasks 30" in source
    assert "--min-rescue 25" in source
    assert "--min-reverse 25" in source
    assert "--max-task-nonzero-share 0.10" in source
    assert "--min-replay-valid-rate 0.90" in source
