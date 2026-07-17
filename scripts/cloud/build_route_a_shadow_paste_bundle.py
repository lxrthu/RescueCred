#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "rescuecredit/appworld_shadow_credit.py",
    "scripts/appworld_azure_continuation_worker.py",
    "scripts/attach_appworld_shadow_credit.py",
    "scripts/check_route_a_shadow_gate.py",
    "scripts/cloud/run_route_a_shadow_smoke.sh",
    "scripts/cloud/run_route_a_shadow_confirm.sh",
    "tests/test_appworld_shadow_credit.py",
    "docs/ROUTE_A_SHADOW_SMOKE_CN.md",
]


def main() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            archive.add(ROOT / relative, arcname=relative)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    output = ROOT / "dist" / "PASTE_ROUTE_A_SHADOW_TO_SERVER.sh"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

python - <<'PY'
import base64
from pathlib import Path

payload = "{encoded}"
target = Path("/tmp/rescuecredit_route_a_shadow.tar.gz")
target.write_bytes(base64.b64decode(payload))
print("WROTE", target)
PY

echo "{digest}  /tmp/rescuecredit_route_a_shadow.tar.gz" | sha256sum -c -
tar -xzf /tmp/rescuecredit_route_a_shadow.tar.gz -C /data/hxy/projects/RescueCredit
chmod +x scripts/cloud/run_route_a_shadow_smoke.sh scripts/cloud/run_route_a_shadow_confirm.sh

source /home/hxy/miniconda3/etc/profile.d/conda.sh
conda activate /data/hxy/venvs/rescuecredit-appworld
unset VIRTUAL_ENV
export APPWORLD_ROOT=/data/hxy/projects/RescueCredit

PY_APPWORLD=/data/hxy/venvs/rescuecredit-appworld/bin/python
"$PY_APPWORLD" -m py_compile \
  rescuecredit/appworld_shadow_credit.py \
  scripts/attach_appworld_shadow_credit.py \
  scripts/check_route_a_shadow_gate.py

OUT=outputs/appworld_route_a_shadow_confirm_seed42
mkdir -p "$OUT"
tmux kill-session -t route_a_shadow_confirm 2>/dev/null || true
tmux new-session -d -s route_a_shadow_confirm \
  "cd /data/hxy/projects/RescueCredit && \
   bash scripts/cloud/run_route_a_shadow_confirm.sh \
   > $OUT/driver.log 2>&1"

sleep 5
tmux ls | grep route_a_shadow_confirm || true
tail -n 30 "$OUT/driver.log"
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"bundle_sha256={digest}")
    print(f"paste_file_bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
