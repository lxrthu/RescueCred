import json

import pytest

from rescuecredit.audit import AuditLedger, UniformAuditScheduler


def test_probability_is_committed_before_draw(tmp_path):
    path = tmp_path / "audit.jsonl"
    ledger = AuditLedger(path)
    commit = ledger.commit("event", 0.2, 0.5)
    draw = ledger.draw("event", 42)
    assert commit.committed_at_ns < draw.drawn_at_ns
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["probability_commit", "audit_draw"]
    assert rows[1]["commit_digest"] == rows[0]["digest"]


def test_draw_without_commit_fails(tmp_path):
    with pytest.raises(RuntimeError):
        AuditLedger(tmp_path / "audit.jsonl").draw("event", 42)


def test_probability_cannot_change_after_commit(tmp_path):
    ledger = AuditLedger(tmp_path / "audit.jsonl")
    ledger.commit("event", 0.2, 0.5)
    with pytest.raises(RuntimeError):
        ledger.commit("event", 0.8, 0.5)


def test_ledger_reloads_and_preserves_commit_state(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = AuditLedger(path)
    first.commit("event", 0.2, 0.5)
    first.draw("event", 42)
    second = AuditLedger(path)
    assert second.committed_probability("event") == 0.2
    with pytest.raises(RuntimeError):
        second.commit("event", 0.3, 0.5)


class _Event:
    def __init__(self, patch_id):
        self.patch_id = patch_id


def test_scheduler_forces_per_patch_warm_start_then_uses_base_probability():
    scheduler = UniformAuditScheduler(0.2, warm_start_events_per_patch=2)
    a = _Event("a")
    b = _Event("b")
    assert [scheduler.probability_for(a, 0.0) for _ in range(3)] == [1.0, 1.0, 0.2]
    assert [scheduler.probability_for(b, 0.0) for _ in range(2)] == [1.0, 1.0]
    assert scheduler.warm_start_assignments == 4


def test_scheduler_rejects_negative_warm_start():
    with pytest.raises(ValueError):
        UniformAuditScheduler(0.2, warm_start_events_per_patch=-1)
