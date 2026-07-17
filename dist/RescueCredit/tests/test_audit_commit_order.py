import json

import pytest

from rescuecredit.audit import AuditLedger


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
