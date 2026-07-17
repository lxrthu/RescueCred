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
    "scripts/prepare_route_a_preference_data.py",
    "scripts/train_route_a_preference.py",
    "scripts/evaluate_route_a_preference.py",
    "scripts/check_route_a_preference_gate.py",
    "scripts/cloud/run_route_a_seed42_preference_pair.sh",
    "tests/test_route_a_preference.py",
    "docs/ROUTE_A_SEED42_PREFERENCE_PILOT_CN.md",
    "refine-logs/ROUTE_A_SEED42_CODE_REVIEW.md",
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
    output = ROOT / "dist" / "PASTE_ROUTE_A_SEED42_TO_SERVER.sh"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

python - <<'PY'
import base64
from pathlib import Path

payload = "{encoded}"
target = Path("/tmp/rescuecredit_route_a_seed42.tar.gz")
target.write_bytes(base64.b64decode(payload))
print("WROTE", target)
PY

echo "{digest}  /tmp/rescuecredit_route_a_seed42.tar.gz" | sha256sum -c -
tar -xzf /tmp/rescuecredit_route_a_seed42.tar.gz -C /data/hxy/projects/RescueCredit
chmod +x scripts/cloud/run_route_a_seed42_preference_pair.sh

source data_disk_env.sh
source .venv/bin/activate
PY=/data/hxy/projects/RescueCredit/.venv/bin/python

"$PY" -m py_compile \\
  rescuecredit/route_a_preference.py \\
  scripts/prepare_route_a_preference_data.py \\
  scripts/train_route_a_preference.py \\
  scripts/evaluate_route_a_preference.py \\
  scripts/check_route_a_preference_gate.py

"$PY" -m pytest -q tests/test_route_a_preference.py

mkdir -p outputs/route_a_seed42_preference_pair
tmux kill-session -t route_a_seed42 2>/dev/null || true
tmux new-session -d -s route_a_seed42 \\
  "cd /data/hxy/projects/RescueCredit && bash scripts/cloud/run_route_a_seed42_preference_pair.sh 2>&1 | tee outputs/route_a_seed42_preference_pair/driver.log"

echo "STARTED tmux=route_a_seed42"
echo "Monitor: tail -f outputs/route_a_seed42_preference_pair/driver.log"
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"bundle_sha256={digest}")
    print(f"paste_file_bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
