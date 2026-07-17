#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "scripts/repair_route_a_v21_balanced_data.py",
    "scripts/prepare_route_a_v21_data.py",
    "scripts/check_route_a_v21_data.py",
    "scripts/cloud/run_route_a_v21c_repair.sh",
    "tests/test_route_a_v21c_repair.py",
]


def main() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            archive.add(ROOT / relative, arcname=relative)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    payload_hash = hashlib.sha256(payload).hexdigest()
    output = ROOT / "dist/PASTE_APPWORLD_V21C_REPAIR_TO_SERVER.sh"
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
target = Path("/tmp/rescuecredit_v21c_repair.tar.gz")
target.write_bytes(base64.b64decode(payload))
print("WROTE", target)
PY

echo "{payload_hash}  /tmp/rescuecredit_v21c_repair.tar.gz" | sha256sum -c -
tar -xzf /tmp/rescuecredit_v21c_repair.tar.gz -C /data/hxy/projects/RescueCredit
chmod +x scripts/cloud/run_route_a_v21c_repair.sh

APP_PY=/data/hxy/venvs/rescuecredit-appworld/bin/python
"$APP_PY" -m py_compile \
  scripts/repair_route_a_v21_balanced_data.py \
  scripts/prepare_route_a_v21_data.py \
  scripts/check_route_a_v21_data.py
"$APP_PY" -m pytest -q tests/test_route_a_v21c_repair.py tests/test_route_a_v21_data.py

bash scripts/cloud/run_route_a_v21c_repair.sh

cat outputs/route_a_v21c_balanced_data_seed42/repair_manifest.json
cat outputs/route_a_v21c_balanced_data_seed42/data_gate.json
)
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"bundle_sha256={payload_hash}")


if __name__ == "__main__":
    main()
