from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


REQUIRED_V4_SOURCE_PATHS = (
    "environments/toolsandbox/__init__.py",
    "environments/toolsandbox/adapter.py",
    "rescuecredit/appworld_shadow_credit.py",
    "rescuecredit/azure_client.py",
    "rescuecredit/toolsandbox_audit.py",
    "rescuecredit/toolsandbox_credit.py",
    "rescuecredit/toolsandbox_protocol.py",
    "scripts/audit_toolsandbox_signal.py",
    "scripts/check_toolsandbox_v41_diagnostic_gate.py",
    "scripts/check_llm.py",
    "scripts/cloud/run_toolsandbox_v4_signal_audit.sh",
    "scripts/cloud/run_toolsandbox_v41_toolid_audit.sh",
    "scripts/freeze_toolsandbox_v4_protocol.py",
    "scripts/toolsandbox_azure_worker.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_vendor_identity(package_root: Path, expected_commit: str) -> Dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(package_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()

    vendor_root = Path(git("rev-parse", "--show-toplevel")).resolve()
    head = git("rev-parse", "HEAD")
    if head != expected_commit:
        raise RuntimeError(
            f"ToolSandbox vendor HEAD mismatch: {head} != {expected_commit}"
        )
    tracked_status = git("status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise RuntimeError("ToolSandbox vendor has tracked worktree modifications")
    return {
        "vendor_git_root": str(vendor_root),
        "vendor_git_head": head,
        "vendor_tracked_worktree_clean": True,
    }


def current_toolsandbox_runtime_identity(expected_commit: str) -> Dict[str, Any]:
    import tool_sandbox

    package_entry = Path(tool_sandbox.__file__).resolve()
    package_root = package_entry.parent
    python = Path(sys.executable).resolve()
    python_sources = sorted(package_root.rglob("*.py"))
    tree_digest = hashlib.sha256()
    for source in python_sources:
        relative = source.relative_to(package_root).as_posix().encode("utf-8")
        tree_digest.update(len(relative).to_bytes(8, "big"))
        tree_digest.update(relative)
        content = source.read_bytes()
        tree_digest.update(len(content).to_bytes(8, "big"))
        tree_digest.update(content)
    return {
        "python": str(python),
        "python_sha256": sha256_file(python),
        "package_entry": str(package_entry),
        "package_entry_sha256": sha256_file(package_entry),
        "package_python_files": len(python_sources),
        "package_python_tree_sha256": tree_digest.hexdigest(),
        **git_vendor_identity(package_root, expected_commit),
    }
