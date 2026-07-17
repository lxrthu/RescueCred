#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "rescuecredit/route_a_preference.py",
    "scripts/train_route_a_preference.py",
    "scripts/evaluate_route_a_preference.py",
    "scripts/freeze_route_a_v3_expanded_protocol.py",
    "scripts/check_route_a_v3_expanded_gate.py",
    "scripts/cloud/run_route_a_v3_expanded_seed42.sh",
    "tests/test_route_a_v3_expanded.py",
]


def main() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            archive.add(ROOT / relative, arcname=relative)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    payload_hash = hashlib.sha256(payload).hexdigest()
    output = ROOT / "dist/PASTE_APPWORLD_V3_EXPANDED_SEED42_TO_SERVER.sh"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''(
set -e
cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

python - <<'PY'
import base64
from pathlib import Path
payload = "{encoded}"
target = Path("/tmp/rescuecredit_v3_expanded_seed42.tar.gz")
target.write_bytes(base64.b64decode(payload))
print("WROTE", target)
PY

echo "{payload_hash}  /tmp/rescuecredit_v3_expanded_seed42.tar.gz" | sha256sum -c -
tar -xzf /tmp/rescuecredit_v3_expanded_seed42.tar.gz -C /data/hxy/projects/RescueCredit
chmod +x scripts/cloud/run_route_a_v3_expanded_seed42.sh

PY=/data/hxy/projects/RescueCredit/.venv/bin/python
"$PY" -m py_compile \
  scripts/train_route_a_preference.py \
  scripts/evaluate_route_a_preference.py \
  scripts/freeze_route_a_v3_expanded_protocol.py \
  scripts/check_route_a_v3_expanded_gate.py
"$PY" -m pytest -q tests/test_route_a_v3_expanded.py tests/test_route_a_preference.py

test -f outputs/route_a_v21c_balanced_data_seed42/data_gate.json
test ! -e outputs/route_a_v3_expanded_seed42
test ! -e outputs/route_a_v3_expanded_seed42.driver.log
tmux has-session -t v3_expanded42 2>/dev/null && tmux kill-session -t v3_expanded42 || true

tmux new-session -d -s v3_expanded42 \
  "cd /data/hxy/projects/RescueCredit && bash scripts/cloud/run_route_a_v3_expanded_seed42.sh 2>&1 | tee outputs/route_a_v3_expanded_seed42.driver.log"

echo "STARTED tmux=v3_expanded42"
echo "Monitor: tail -f outputs/route_a_v3_expanded_seed42.driver.log"
echo "Final: cat outputs/route_a_v3_expanded_seed42/gate.json"
)
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"bundle_sha256={payload_hash}")


if __name__ == "__main__":
    main()
