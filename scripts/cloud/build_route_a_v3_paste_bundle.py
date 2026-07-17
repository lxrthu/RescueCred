#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import tarfile
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "rescuecredit/frozen_bank.py",
    "rescuecredit/route_a_preference.py",
    "scripts/train_route_a_preference.py",
    "scripts/evaluate_route_a_preference.py",
    "scripts/check_route_a_v3_gate.py",
    "scripts/freeze_route_a_v3_protocol.py",
    "scripts/cloud/run_route_a_v3_absolute_seed42.sh",
    "tests/test_route_a_preference.py",
    "tests/test_route_a_signal_hotfix.py",
    "tests/test_route_a_v3_gate.py",
    "tests/test_route_a_v3_protocol.py",
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
    digest = hashlib.sha256(payload).hexdigest()
    encoded = "\n".join(textwrap.wrap(base64.b64encode(payload).decode("ascii"), 76))
    output = ROOT / "dist" / "PASTE_ROUTE_A_V3_TO_SERVER.sh"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

base64 -d > /tmp/rescuecredit_route_a_v3.tar.gz <<'PAYLOAD'
{encoded}
PAYLOAD

echo "{digest}  /tmp/rescuecredit_route_a_v3.tar.gz" | sha256sum -c -
tar -xzf /tmp/rescuecredit_route_a_v3.tar.gz -C /data/hxy/projects/RescueCredit
chmod +x scripts/cloud/run_route_a_v3_absolute_seed42.sh

PY=/data/hxy/projects/RescueCredit/.venv/bin/python
"$PY" -m py_compile \\
  rescuecredit/frozen_bank.py \\
  rescuecredit/route_a_preference.py \\
  scripts/train_route_a_preference.py \\
  scripts/evaluate_route_a_preference.py \\
  scripts/check_route_a_v3_gate.py \\
  scripts/freeze_route_a_v3_protocol.py
"$PY" -m pytest -q \\
  tests/test_route_a_signal_hotfix.py \\
  tests/test_route_a_preference.py \\
  tests/test_route_a_v3_gate.py \\
  tests/test_route_a_v3_protocol.py

OUT=outputs/route_a_v3_absolute_seed42
mkdir -p "$OUT"
tmux kill-session -t route_a_v3_abs42 2>/dev/null || true
tmux new-session -d -s route_a_v3_abs42 \\
  "cd /data/hxy/projects/RescueCredit && bash scripts/cloud/run_route_a_v3_absolute_seed42.sh 2>&1 | tee $OUT/driver.log"

echo "STARTED tmux=route_a_v3_abs42"
echo "Monitor: tail -f $OUT/driver.log"
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"bundle_sha256={digest}")
    print(f"paste_file_bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
