from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditCommit:
    event_id: str
    probability: float
    mu: float
    committed_at_ns: int
    digest: str


@dataclass(frozen=True)
class AuditDraw:
    event_id: str
    draw: int
    drawn_at_ns: int
    commit_digest: str


class AuditLedger:
    """Append-only commit-before-draw ledger with tamper checks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._commits: dict[str, AuditCommit] = {}
        self._draws: dict[str, AuditDraw] = {}
        if self.path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            event_id = record["event_id"]
            if record["kind"] == "probability_commit":
                if event_id in self._commits:
                    raise RuntimeError(f"duplicate commit in ledger at line {line_number}")
                commit = AuditCommit(
                    event_id=event_id,
                    probability=float(record["probability"]),
                    mu=float(record["mu"]),
                    committed_at_ns=int(record["committed_at_ns"]),
                    digest=record["digest"],
                )
                payload = f"{event_id}|{commit.probability:.17g}|{commit.mu:.17g}|{commit.committed_at_ns}"
                if hashlib.sha256(payload.encode()).hexdigest() != commit.digest:
                    raise RuntimeError(f"invalid commit digest at line {line_number}")
                self._commits[event_id] = commit
            elif record["kind"] == "audit_draw":
                if event_id not in self._commits or event_id in self._draws:
                    raise RuntimeError(f"draw without unique prior commit at line {line_number}")
                draw = AuditDraw(
                    event_id=event_id,
                    draw=int(record["draw"]),
                    drawn_at_ns=int(record["drawn_at_ns"]),
                    commit_digest=record["commit_digest"],
                )
                commit = self._commits[event_id]
                if draw.commit_digest != commit.digest or draw.drawn_at_ns <= commit.committed_at_ns:
                    raise RuntimeError(f"invalid draw linkage/order at line {line_number}")
                self._draws[event_id] = draw
            else:
                raise RuntimeError(f"unknown audit ledger record at line {line_number}")

    def commit(self, event_id: str, probability: float, mu: float) -> AuditCommit:
        if event_id in self._commits:
            raise RuntimeError(f"probability already committed for {event_id}")
        if not 0.0 < probability <= 1.0:
            raise ValueError("probability must be in (0, 1]")
        timestamp = time.time_ns()
        payload = f"{event_id}|{probability:.17g}|{mu:.17g}|{timestamp}"
        record = AuditCommit(event_id, float(probability), float(mu), timestamp, hashlib.sha256(payload.encode()).hexdigest())
        self._commits[event_id] = record
        self._append({"kind": "probability_commit", **record.__dict__})
        return record

    def draw(self, event_id: str, seed: int) -> AuditDraw:
        if event_id not in self._commits:
            raise RuntimeError("audit probability must be committed before draw")
        if event_id in self._draws:
            raise RuntimeError(f"draw already recorded for {event_id}")
        commit = self._commits[event_id]
        draw = int(random.Random(seed).random() < commit.probability)
        timestamp = max(time.time_ns(), commit.committed_at_ns + 1)
        record = AuditDraw(event_id, draw, timestamp, commit.digest)
        self._draws[event_id] = record
        self._append({"kind": "audit_draw", **record.__dict__})
        return record

    def committed_probability(self, event_id: str) -> float:
        return self._commits[event_id].probability

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


class UniformAuditScheduler:
    """Uniform auditing with an exact per-patch warm start.

    Warm-start events are committed with probability one, so their exact
    counterfactuals initialize the patch EMA without bias. Later events use
    the configured Bernoulli probability.
    """

    def __init__(self, probability: float, warm_start_events_per_patch: int = 0) -> None:
        if not 0.0 < probability <= 1.0:
            raise ValueError("probability must be in (0, 1]")
        if warm_start_events_per_patch < 0:
            raise ValueError("warm_start_events_per_patch must be non-negative")
        self.probability = float(probability)
        self.warm_start_events_per_patch = int(warm_start_events_per_patch)
        self._seen_by_patch: dict[str, int] = {}
        self.warm_start_assignments = 0

    def probability_for(self, event: object, _mu: float) -> float:
        patch_id = getattr(event, "patch_id", None)
        if not isinstance(patch_id, str) or not patch_id:
            raise ValueError("audit event must expose a non-empty patch_id")
        seen = self._seen_by_patch.get(patch_id, 0)
        self._seen_by_patch[patch_id] = seen + 1
        if seen < self.warm_start_events_per_patch:
            self.warm_start_assignments += 1
            return 1.0
        return self.probability
