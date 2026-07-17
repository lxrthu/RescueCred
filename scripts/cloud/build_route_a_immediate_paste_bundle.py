#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "rescuecredit/route_a_immediate.py",
    "scripts/evaluate_route_a_immediate.py",
    "scripts/check_route_a_immediate_gate.py",
    "scripts/cloud/run_route_a_appworld_immediate.sh",
    "tests/test_route_a_immediate.py",
    "docs/ROUTE_A_APPWORLD_IMMEDIATE_CN.md",
    "refine-logs/ROUTE_A_IMMEDIATE_CODE_REVIEW_20260716_201801.md",
    "refine-logs/ROUTE_A_IMMEDIATE_CODE_REVIEW.md",
    "refine-logs/ROUTE_A_IMMEDIATE_TRACKER_20260716_201801.md",
    "refine-logs/ROUTE_A_IMMEDIATE_TRACKER.md",
]


def main() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.add(path, arcname=relative)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    output = ROOT / "dist" / "PASTE_ROUTE_A_IMMEDIATE_TO_SERVER.sh"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

python - <<'PY'
import base64
from pathlib import Path

payload = "{encoded}"
target = Path("/tmp/rescuecredit_route_a_immediate.tar.gz")
target.write_bytes(base64.b64decode(payload))
print("WROTE", target)
PY

echo "{digest}  /tmp/rescuecredit_route_a_immediate.tar.gz" | sha256sum -c -
tar -xzf /tmp/rescuecredit_route_a_immediate.tar.gz -C /data/hxy/projects/RescueCredit
chmod +x scripts/cloud/run_route_a_appworld_immediate.sh

MODEL_PY=/data/hxy/projects/RescueCredit/.venv/bin/python
APP_PY=/data/hxy/venvs/rescuecredit-appworld/bin/python

"$MODEL_PY" -m py_compile \\
  rescuecredit/route_a_immediate.py \\
  scripts/evaluate_route_a_immediate.py \\
  scripts/check_route_a_immediate_gate.py
"$MODEL_PY" -m pytest -q tests/test_route_a_immediate.py
"$APP_PY" -m py_compile scripts/evaluate_route_a_immediate.py

test -f outputs/route_a_appworld_dev_seed42_v2/events/dev_events.public.jsonl
test -f outputs/route_a_appworld_dev_seed42_v2/mask/task_results.jsonl
test -f outputs/route_a_appworld_dev_seed42_v2/v2/task_results.jsonl

mkdir -p outputs/route_a_appworld_dev_immediate_seed42
tmux kill-session -t route_a_immediate42 2>/dev/null || true
tmux new-session -d -s route_a_immediate42 \\
  "cd /data/hxy/projects/RescueCredit && bash scripts/cloud/run_route_a_appworld_immediate.sh 2>&1 | tee outputs/route_a_appworld_dev_immediate_seed42/driver.log"

echo "STARTED tmux=route_a_immediate42"
echo "No GPU or Azure API is used."
echo "Monitor: tail -f outputs/route_a_appworld_dev_immediate_seed42/driver.log"
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"bundle_sha256={digest}")
    print(f"paste_file_bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
