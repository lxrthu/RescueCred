#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/attach_appworld_shadow_credit_v21.py"
OUTPUT = ROOT / "dist/PASTE_APPWORLD_V21_SHADOW_IMPORT_HOTFIX_TO_SERVER.sh"


def main() -> None:
    payload = SOURCE.read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    payload_hash = hashlib.sha256(payload).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        f'''(
set -e
cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

python - <<'PY'
import base64
from pathlib import Path

payload = "{encoded}"
target = Path("scripts/attach_appworld_shadow_credit_v21.py")
target.write_bytes(base64.b64decode(payload))
print("PATCHED", target)
PY

echo "{payload_hash}  scripts/attach_appworld_shadow_credit_v21.py" | sha256sum -c -

APP_PY=/data/hxy/venvs/rescuecredit-appworld/bin/python
"$APP_PY" -m py_compile scripts/attach_appworld_shadow_credit_v21.py
"$APP_PY" - <<'PY'
import sys
sys.path.insert(0, "scripts")
import attach_appworld_shadow_credit_v21 as module
assert issubclass(module.WorkerFatalError, RuntimeError)
assert callable(module._run_branch)
print("V21_SHADOW_IMPORT_HOTFIX_OK")
PY

OUT=outputs/route_a_v21b_expanded_data_seed42
DRIVER=outputs/route_a_v21b_data_expansion.driver.log
test -f "$OUT/bank/manifest.json"

if tmux has-session -t v21b_data 2>/dev/null; then
  tmux kill-session -t v21b_data
fi

tmux new-session -d -s v21b_data \
  "cd /data/hxy/projects/RescueCredit && RESUME=1 bash scripts/cloud/run_route_a_v21_data_expansion.sh 2>&1 | tee -a $DRIVER"

echo "RESUMED tmux=v21b_data from the existing frozen bank"
echo "Monitor: tail -f $DRIVER"
echo "Final: cat $OUT/data_gate.json"
)
''',
        encoding="utf-8",
        newline="\n",
    )
    print(OUTPUT)
    print(f"source_sha256={payload_hash}")
    print(f"paste_file_bytes={OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
