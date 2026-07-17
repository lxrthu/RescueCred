#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "environments/appworld/__init__.py",
    "environments/appworld/adapter.py",
    "scripts/inspect_appworld_contract.py",
    "scripts/cloud/setup_appworld_stage0.sh",
    "tests/test_appworld_adapter.py",
    "docs/APPWORLD_STAGE0_CN.md",
    "refine-logs/APPWORLD_MIGRATION_PLAN.md",
    "refine-logs/APPWORLD_STAGE0_CODE_REVIEW.md",
    "pyproject.toml",
]


def main() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            archive.add(ROOT / relative, arcname=relative)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()

    output = ROOT / "dist" / "PASTE_APPWORLD_STAGE0_TO_SERVER.sh"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''cd /data/hxy/projects/RescueCredit

python - <<'PY'
import base64
from pathlib import Path

payload = "{encoded}"
Path("/tmp/rescuecredit_appworld_stage0.tar.gz").write_bytes(base64.b64decode(payload))
print("APPWORLD_STAGE0_BUNDLE_WRITTEN")
PY

echo "{digest}  /tmp/rescuecredit_appworld_stage0.tar.gz" | sha256sum -c -
tar -xzf /tmp/rescuecredit_appworld_stage0.tar.gz -C /data/hxy/projects/RescueCredit

source data_disk_env.sh
source .venv/bin/activate
python -m py_compile \
  environments/appworld/__init__.py \
  environments/appworld/adapter.py \
  scripts/inspect_appworld_contract.py \
  tests/test_appworld_adapter.py
pytest -q tests/test_appworld_adapter.py

chmod +x scripts/cloud/setup_appworld_stage0.sh
tmux kill-session -t appworld_stage0 2>/dev/null || true
tmux new-session -d -s appworld_stage0 \
  "cd /data/hxy/projects/RescueCredit && bash scripts/cloud/setup_appworld_stage0.sh 2>&1 | tee outputs/appworld_stage0_console.log"

sleep 5
tail -n 40 outputs/appworld_stage0_console.log
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"bundle_sha256={digest}")
    print(f"paste_file_bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
