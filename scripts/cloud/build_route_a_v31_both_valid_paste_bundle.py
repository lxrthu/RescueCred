#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "scripts/build_route_a_both_valid_dev_events.py",
    "scripts/audit_appworld_deployable_harness.py",
    "scripts/select_route_a_frozen_events.py",
    "scripts/evaluate_route_a_bounded.py",
    "scripts/freeze_route_a_v31_both_valid_protocol.py",
    "scripts/audit_route_a_v31_both_valid_bounded.py",
    "scripts/cloud/run_route_a_v31_both_valid_dev_seed42.sh",
    "rescuecredit/route_a_bounded.py",
    "tests/test_route_a_v31_both_valid.py",
)
OUTPUT = ROOT / "dist/PASTE_APPWORLD_V31_BOTH_VALID_DEV_TO_SERVER.sh"


def main() -> None:
    payloads = {
        name: base64.b64encode((ROOT / name).read_bytes()).decode("ascii")
        for name in SOURCES
    }
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCES
    }
    files_literal = repr(list(SOURCES))
    writes = "\n".join(
        f"Path({name!r}).write_bytes(base64.b64decode({payload!r}))"
        for name, payload in payloads.items()
    )
    checks = "\n".join(
        f'echo "{digest}  {name}" | sha256sum -c -'
        for name, digest in hashes.items()
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        f'''(
set -e
cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

python - <<'PY'
import base64
import shutil
import time
from pathlib import Path

files = {files_literal}
backup = Path("outputs/code_backups") / f"v31_both_valid_{{time.strftime('%Y%m%d_%H%M%S')}}"
for name in files:
    source = Path(name)
    if source.is_file():
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
{writes}
print(f"WROTE V3.1 both-valid AppWorld dev patch; backup={{backup}}")
PY

{checks}

PY=/data/hxy/projects/RescueCredit/.venv/bin/python
"$PY" -m py_compile \
  scripts/build_route_a_both_valid_dev_events.py \
  scripts/select_route_a_frozen_events.py \
  scripts/evaluate_route_a_bounded.py \
  scripts/freeze_route_a_v31_both_valid_protocol.py \
  scripts/audit_route_a_v31_both_valid_bounded.py
"$PY" -m pytest -q \
  tests/test_route_a_v31_both_valid.py \
  tests/test_route_a_bounded.py \
  tests/test_route_a_bounded_contract.py
bash -n scripts/cloud/run_route_a_v31_both_valid_dev_seed42.sh
chmod +x scripts/cloud/run_route_a_v31_both_valid_dev_seed42.sh

STAMP=$(date +%Y%m%d_%H%M%S)
if [ -e outputs/route_a_v31_both_valid_appworld_dev_seed42 ]; then
  mv outputs/route_a_v31_both_valid_appworld_dev_seed42 \
     outputs/route_a_v31_both_valid_appworld_dev_seed42.previous_$STAMP
fi
tmux kill-session -t route_a_v31_bv42 2>/dev/null || true
tmux new-session -d -s route_a_v31_bv42 \
  "cd /data/hxy/projects/RescueCredit && bash scripts/cloud/run_route_a_v31_both_valid_dev_seed42.sh 2>&1 | tee outputs/route_a_v31_both_valid_appworld_dev_seed42.driver.log"

sleep 3
tmux ls | grep route_a_v31_bv42
tail -n 30 outputs/route_a_v31_both_valid_appworld_dev_seed42.driver.log || true
echo "Monitor: tail -f outputs/route_a_v31_both_valid_appworld_dev_seed42.driver.log"
)
''',
        encoding="utf-8",
        newline="\n",
    )
    print(OUTPUT)
    print(f"paste_file_sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
