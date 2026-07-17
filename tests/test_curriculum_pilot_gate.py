from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_gate(tmp_path: Path, rescue_sequence: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    mask_eval = _write(tmp_path / "mask_eval.json", {"s_off": 0.2, "first_pass": 0.4})
    rescue_eval = _write(tmp_path / "rescue_eval.json", {"s_off": 0.3, "first_pass": 0.4})
    common_sampling = {
        "visible_curriculum_fraction": 0.75,
        "visible_pool_hash": "pool",
        "assignment_sequence_hash": "sequence",
    }
    mask_run = _write(
        tmp_path / "mask_run.json",
        {
            "split_hash": "split",
            "world_size": 2,
            "main_interaction_budget": 2000,
            "sampling": common_sampling,
        },
    )
    rescue_sampling = dict(common_sampling, assignment_sequence_hash=rescue_sequence)
    rescue_run = _write(
        tmp_path / "rescue_run.json",
        {
            "split_hash": "split",
            "world_size": 2,
            "main_interaction_budget": 2000,
            "sampling": rescue_sampling,
            "audit_stats": {"valid_audits": 10},
        },
    )
    output = tmp_path / "gate.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_pilot_gate.py"),
            "--mask",
            str(mask_eval),
            "--rescue",
            str(rescue_eval),
            "--mask-run-summary",
            str(mask_run),
            "--rescue-run-summary",
            str(rescue_run),
            "--min-audited-events",
            "8",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return result, json.loads(output.read_text(encoding="utf-8"))


def test_gate_accepts_identical_curriculum_assignments(tmp_path):
    result, gate = _run_gate(tmp_path, rescue_sequence="sequence")
    assert result.returncode == 0
    assert gate["passed"] is True
    assert gate["sampling_gate"] == {"checked": True, "passed": True, "mismatches": {}}


def test_gate_rejects_different_curriculum_assignments(tmp_path):
    result, gate = _run_gate(tmp_path, rescue_sequence="different")
    assert result.returncode == 2
    assert gate["passed"] is False
    assert gate["sampling_gate"]["passed"] is False
    assert "assignment_sequence_hash" in gate["sampling_gate"]["mismatches"]
