#!/usr/bin/env python3
from __future__ import annotations

import base64
import gzip
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "probe_appworld_rollback.py"


def main() -> None:
    payload = gzip.compress(SOURCE.read_bytes(), compresslevel=9)
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    output = ROOT / "dist" / "PASTE_APPWORLD_AW1_TO_SERVER.sh"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

python - <<'PY'
import base64
import gzip
from pathlib import Path

payload = "{encoded}"
target = Path("scripts/probe_appworld_rollback.py")
target.write_bytes(gzip.decompress(base64.b64decode(payload)))
print("WROTE", target)
PY

python -m py_compile scripts/probe_appworld_rollback.py

mkdir -p outputs/appworld_rollback_probe
PY=/data/hxy/venvs/rescuecredit-appworld/bin/python
APPWORLD_ROOT=/data/hxy/projects/RescueCredit \
"$PY" scripts/probe_appworld_rollback.py \
  --appworld-root /data/hxy/projects/RescueCredit \
  --subset train \
  --limit 30 \
  --minimum-rollbacks 3 \
  --output-dir outputs/appworld_rollback_probe \
  > outputs/appworld_rollback_probe/console.log 2>&1

echo "AW1_EXIT=$?"
tail -n 100 outputs/appworld_rollback_probe/console.log
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"bundle_sha256={digest}")
    print(f"paste_file_bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
