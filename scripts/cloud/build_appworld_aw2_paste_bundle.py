#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "environments/appworld/deployable.py",
    "scripts/appworld_candidate_selector_worker.py",
    "scripts/appworld_azure_candidate_selector_worker.py",
    "scripts/audit_appworld_deployable_harness.py",
    "scripts/cloud/run_appworld_aw2h_provenance.sh",
    "tests/test_appworld_deployable.py",
]


def main() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            archive.add(ROOT / relative, arcname=relative)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    output = ROOT / "dist" / "PASTE_APPWORLD_AW2_TO_SERVER.sh"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

python - <<'PY'
import base64
from pathlib import Path

payload = "{encoded}"
target = Path("/tmp/rescuecredit_appworld_aw2h.tar.gz")
target.write_bytes(base64.b64decode(payload))
print("WROTE", target)
PY

echo "{digest}  /tmp/rescuecredit_appworld_aw2h.tar.gz" | sha256sum -c -
tar -xzf /tmp/rescuecredit_appworld_aw2h.tar.gz -C /data/hxy/projects/RescueCredit

PY_APPWORLD=/data/hxy/venvs/rescuecredit-appworld/bin/python
"$PY_APPWORLD" -m py_compile \
  environments/appworld/deployable.py \
  scripts/appworld_candidate_selector_worker.py \
  scripts/appworld_azure_candidate_selector_worker.py \
  scripts/audit_appworld_deployable_harness.py

FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
  awk -F, '{{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2 < 500) {{print $1; exit}}}}')
if [ -z "$FREE_GPU" ]; then
  FREE_GPU=0
fi
echo "AW2H_DOES_NOT_REQUIRE_GPU"

OUT=outputs/appworld_harness_audit_30_v8_provenance
mkdir -p "$OUT"
tmux kill-session -t appworld_aw2h 2>/dev/null || true
tmux new-session -d -s appworld_aw2h \
  "cd /data/hxy/projects/RescueCredit && \
   bash scripts/cloud/run_appworld_aw2h_provenance.sh \
     > $OUT/driver.log 2>&1"

sleep 5
tmux ls | grep appworld_aw2h || true
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
