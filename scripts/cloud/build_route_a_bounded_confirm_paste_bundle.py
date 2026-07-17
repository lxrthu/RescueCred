#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "rescuecredit/route_a_bounded.py",
    "rescuecredit/appworld_shadow_credit.py",
    "rescuecredit/azure_client.py",
    "environments/appworld/adapter.py",
    "scripts/evaluate_route_a_bounded.py",
    "scripts/freeze_route_a_bounded_confirm_protocol.py",
    "scripts/analyze_route_a_bounded_confirm.py",
    "scripts/appworld_azure_continuation_worker.py",
    "scripts/attach_appworld_shadow_credit.py",
    "scripts/audit_appworld_deployable_harness.py",
    "scripts/cloud/run_route_a_appworld_bounded_confirm.sh",
    "tests/test_route_a_bounded_confirm.py",
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
    output = ROOT / "dist" / "PASTE_APPWORLD_BOUNDED_CONFIRM_TO_SERVER.sh"
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
target = Path("/tmp/rescuecredit_appworld_bounded_confirm.tar.gz")
target.write_bytes(base64.b64decode(payload))
print("WROTE", target)
PY

echo "{digest}  /tmp/rescuecredit_appworld_bounded_confirm.tar.gz" | sha256sum -c -
tar -xzf /tmp/rescuecredit_appworld_bounded_confirm.tar.gz -C /data/hxy/projects/RescueCredit
chmod +x scripts/cloud/run_route_a_appworld_bounded_confirm.sh

MODEL_PY=/data/hxy/projects/RescueCredit/.venv/bin/python
APP_PY=/data/hxy/venvs/rescuecredit-appworld/bin/python

"$MODEL_PY" -m py_compile \
  scripts/evaluate_route_a_bounded.py \
  scripts/freeze_route_a_bounded_confirm_protocol.py \
  scripts/analyze_route_a_bounded_confirm.py
"$APP_PY" -m py_compile scripts/evaluate_route_a_bounded.py
"$MODEL_PY" -m pytest -q \
  tests/test_route_a_bounded.py \
  tests/test_route_a_bounded_contract.py \
  tests/test_route_a_bounded_confirm.py

test -f outputs/route_a_appworld_dev_seed42_v2/events/dev_events.public.jsonl
test -f outputs/route_a_appworld_dev_seed42_v2/mask/task_results.jsonl
test -f outputs/route_a_appworld_dev_seed42_v2/v2/task_results.jsonl
test -f .env

OUT=outputs/route_a_appworld_bounded_confirm_43_44_45
DRIVER=outputs/route_a_appworld_bounded_confirm.driver.log
if [ -e "$OUT" ] || [ -e "$DRIVER" ]; then
  echo "Refusing to overwrite an existing confirmatory run." >&2
  echo "Existing path: $OUT or $DRIVER" >&2
  false
fi
if tmux has-session -t appworld_confirm 2>/dev/null; then
  echo "tmux session appworld_confirm already exists" >&2
  false
fi

tmux new-session -d -s appworld_confirm \
  "cd /data/hxy/projects/RescueCredit && bash scripts/cloud/run_route_a_appworld_bounded_confirm.sh 2>&1 | tee $DRIVER"

echo "STARTED tmux=appworld_confirm"
echo "This runs seeds 43/44/45 in parallel through Azure; local GPUs are not used."
echo "Monitor: tail -f $DRIVER"
echo "Final: cat $OUT/combined_gate.json"
)
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"bundle_sha256={digest}")
    print(f"paste_file_bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
