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
    "rescuecredit/appworld_shadow_credit.py",
    "rescuecredit/azure_client.py",
    "rescuecredit/frozen_bank.py",
    "scripts/build_appworld_route_a_bank_v21.py",
    "scripts/attach_appworld_shadow_credit_v21.py",
    "scripts/prepare_route_a_v21_data.py",
    "scripts/check_route_a_v21_data.py",
    "scripts/attach_appworld_shadow_credit.py",
    "scripts/recompute_route_a_dense_credit.py",
    "scripts/check_route_a_bank.py",
    "scripts/appworld_azure_continuation_worker.py",
    "scripts/appworld_azure_candidate_selector_worker.py",
    "scripts/audit_appworld_deployable_harness.py",
    "scripts/cloud/run_route_a_v21_data_expansion.sh",
    "tests/test_route_a_v21_data.py",
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
    payload_hash = hashlib.sha256(payload).hexdigest()
    output = ROOT / "dist" / "PASTE_APPWORLD_V21_DATA_EXPANSION_TO_SERVER.sh"
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
target = Path("/tmp/rescuecredit_v21_data_expansion.tar.gz")
target.write_bytes(base64.b64decode(payload))
print("WROTE", target)
PY

echo "{payload_hash}  /tmp/rescuecredit_v21_data_expansion.tar.gz" | sha256sum -c -
tar -xzf /tmp/rescuecredit_v21_data_expansion.tar.gz -C /data/hxy/projects/RescueCredit
chmod +x scripts/cloud/run_route_a_v21_data_expansion.sh

APP_PY=/data/hxy/venvs/rescuecredit-appworld/bin/python
MODEL_PY=/data/hxy/projects/RescueCredit/.venv/bin/python
"$APP_PY" -m py_compile \
  scripts/build_appworld_route_a_bank_v21.py \
  scripts/attach_appworld_shadow_credit_v21.py \
  scripts/prepare_route_a_v21_data.py \
  scripts/check_route_a_v21_data.py
"$MODEL_PY" -m py_compile scripts/appworld_azure_continuation_worker.py
"$MODEL_PY" -m pytest -q tests/test_route_a_v21_data.py

test -f .env
test -d data/tasks
test -f data/datasets/train.txt
test -d /data/hxy/lxr/truth-is-not-enough/models/Qwen2.5-7B-Instruct

OUT=outputs/route_a_v21b_expanded_data_seed42
DRIVER=outputs/route_a_v21b_data_expansion.driver.log
if [ -e "$OUT" ] || [ -e "$DRIVER" ]; then
  echo "Refusing to overwrite an existing V2.1 data run." >&2
  echo "For an interrupted run, use RESUME=1 with the runner after inspecting logs." >&2
  false
fi
if tmux has-session -t v21b_data 2>/dev/null; then
  echo "tmux session v21b_data already exists" >&2
  false
fi

tmux new-session -d -s v21b_data \
  "cd /data/hxy/projects/RescueCredit && bash scripts/cloud/run_route_a_v21_data_expansion.sh 2>&1 | tee $DRIVER"

echo "STARTED tmux=v21b_data"
echo "This is train-only AppWorld data construction and uses Azure, not local GPUs."
echo "Monitor: tail -f $DRIVER"
echo "Final: cat $OUT/data_gate.json"
echo "Resume after interruption: RESUME=1 bash scripts/cloud/run_route_a_v21_data_expansion.sh"
)
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"bundle_sha256={payload_hash}")
    print(f"paste_file_bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
