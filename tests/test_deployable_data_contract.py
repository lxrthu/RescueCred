from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from environments.api_bank.data import digest_records


ROOT = Path(__file__).resolve().parents[1]


def _task(certified: bool) -> dict:
    return {
        "task_id": "contract-task",
        "user_goal": "Cancel appointment 1234567890.",
        "available_tools": [],
        "available_tools_reference_independent": certified,
        "max_steps": 4,
    }


def _run_dry(tmp_path: Path, tasks: list[dict], manifest: dict) -> subprocess.CompletedProcess[str]:
    train_file = tmp_path / "train.jsonl"
    manifest_file = tmp_path / "manifest.json"
    train_file.write_text("\n".join(json.dumps(task) for task in tasks) + "\n", encoding="utf-8")
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_train.py"),
            "--method",
            "rescuecredit_v2",
            "--model",
            "unused",
            "--train-file",
            str(train_file),
            "--manifest",
            str(manifest_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_certified_manifest_cannot_be_paired_with_uncertified_tasks(tmp_path):
    tasks = [_task(certified=False)]
    manifest = {
        "split_hashes": {"train": digest_records(tasks)},
        "available_tools_contract": {"all_runtime_tool_sets_reference_independent": True},
    }
    result = _run_dry(tmp_path, tasks, manifest)
    assert result.returncode != 0
    assert "uncertified available_tools" in result.stderr


def test_manifest_split_hash_must_match_task_file(tmp_path):
    tasks = [_task(certified=True)]
    manifest = {
        "split_hashes": {"train": "not-the-task-digest"},
        "available_tools_contract": {"all_runtime_tool_sets_reference_independent": True},
    }
    result = _run_dry(tmp_path, tasks, manifest)
    assert result.returncode != 0
    assert "digest does not match" in result.stderr
