import subprocess
from pathlib import Path

import pytest

from rescuecredit.toolsandbox_protocol import git_vendor_identity


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def test_vendor_identity_rejects_modified_non_init_runtime_source(tmp_path):
    package = tmp_path / "tool_sandbox"
    package.mkdir()
    (package / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    evaluation = package / "evaluation.py"
    evaluation.write_text("SCORE = 1\n", encoding="utf-8")
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "add", "tool_sandbox")
    _git(tmp_path, "commit", "-m", "pinned runtime")
    head = _git(tmp_path, "rev-parse", "HEAD")

    identity = git_vendor_identity(package, head)
    assert identity["vendor_git_head"] == head
    assert identity["vendor_tracked_worktree_clean"] is True

    evaluation.write_text("SCORE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked worktree modifications"):
        git_vendor_identity(package, head)
