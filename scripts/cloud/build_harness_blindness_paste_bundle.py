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
    "configs/accelerate_h200.yaml",
    "environments/__init__.py",
    "environments/api_bank/__init__.py",
    "environments/api_bank/adapter.py",
    "environments/api_bank/correction_generator.py",
    "environments/api_bank/data.py",
    "environments/api_bank/deployable.py",
    "environments/api_bank/harness.py",
    "environments/api_bank/shadow.py",
    "environments/api_bank/verifier.py",
    "rescuecredit/accounting.py",
    "rescuecredit/__init__.py",
    "rescuecredit/audit.py",
    "rescuecredit/correction_preference.py",
    "rescuecredit/engine.py",
    "rescuecredit/estimators.py",
    "rescuecredit/frozen_bank.py",
    "rescuecredit/logging.py",
    "rescuecredit/training.py",
    "rescuecredit/types.py",
    "rescuecredit/visible_curriculum.py",
    "rescuecredit/v2_preference.py",
    "scripts/run_train.py",
    "scripts/run_eval.py",
    "scripts/freeze_harness_blindness_protocol.py",
    "scripts/analyze_harness_blindness.py",
    "scripts/cloud/run_harness_blindness_seed42.sh",
    "tests/test_harness_blindness.py",
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
    encoded = "\n".join(
        textwrap.wrap(base64.b64encode(payload).decode("ascii"), 76)
    )
    output = ROOT / "dist" / "PASTE_HARNESS_BLINDNESS_TO_SERVER.sh"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

base64 -d > /tmp/rescuecredit_harness_blindness.tar.gz <<'PAYLOAD'
{encoded}
PAYLOAD

echo "{digest}  /tmp/rescuecredit_harness_blindness.tar.gz" | sha256sum -c -
tar -xzf /tmp/rescuecredit_harness_blindness.tar.gz -C /data/hxy/projects/RescueCredit
chmod +x scripts/cloud/run_harness_blindness_seed42.sh

PY=/data/hxy/projects/RescueCredit/.venv/bin/python
"$PY" -m py_compile \\
  environments/api_bank/correction_generator.py \\
  rescuecredit/frozen_bank.py \\
  scripts/run_train.py \\
  scripts/run_eval.py \\
  scripts/freeze_harness_blindness_protocol.py \\
  scripts/analyze_harness_blindness.py
"$PY" -m pytest -q \\
  tests/test_harness_blindness.py \\
  tests/test_training_credit.py \\
  tests/test_snapshot_replay.py

OUT=outputs/harness_credit_blindness_seed42
DRIVER=outputs/harness_credit_blindness_seed42.driver.log
if [ -e "$OUT" ] || [ -e "$DRIVER" ]; then
  echo "Refusing to overwrite existing $OUT or $DRIVER" >&2
  exit 1
fi
if tmux has-session -t harness_blind42 2>/dev/null; then
  echo "tmux session harness_blind42 already exists" >&2
  exit 1
fi
tmux new-session -d -s harness_blind42 \\
  "cd /data/hxy/projects/RescueCredit && bash scripts/cloud/run_harness_blindness_seed42.sh 2>&1 | tee $DRIVER"

echo "STARTED tmux=harness_blind42"
echo "Monitor: tail -f $DRIVER"
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"bundle_sha256={digest}")
    print(f"paste_file_bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
