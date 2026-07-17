#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "environments/api_bank/data.py",
    "rescuecredit/visible_curriculum.py",
    "scripts/prepare_api_bank_controlled.py",
    "scripts/run_train.py",
    "scripts/run_eval.py",
    "scripts/check_pilot_gate.py",
    "scripts/cloud/run_v2_visible_curriculum_smoke_2gpu.sh",
    "tests/test_api_bank_data_receipts.py",
    "tests/test_visible_curriculum.py",
    "tests/test_deployable_data_contract.py",
    "tests/test_curriculum_pilot_gate.py",
]


def main() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            archive.add(ROOT / relative, arcname=relative)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()

    output = ROOT / "dist" / "PASTE_VISIBLE_CURRICULUM_HOTFIX_TO_SERVER.sh"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''cd /data/hxy/projects/RescueCredit

python - <<'PY'
import base64
from pathlib import Path

payload = "{encoded}"
Path("/tmp/rescuecredit_visible_curriculum_hotfix.tar.gz").write_bytes(base64.b64decode(payload))
print("BUNDLE_WRITTEN")
PY

echo "{digest}  /tmp/rescuecredit_visible_curriculum_hotfix.tar.gz" | sha256sum -c -
tar -xzf /tmp/rescuecredit_visible_curriculum_hotfix.tar.gz -C /data/hxy/projects/RescueCredit

source data_disk_env.sh
source .venv/bin/activate

python -m py_compile \\
  environments/api_bank/data.py \\
  rescuecredit/visible_curriculum.py \\
  scripts/prepare_api_bank_controlled.py \\
  scripts/run_train.py \\
  scripts/run_eval.py \\
  scripts/check_pilot_gate.py

pytest -q

python scripts/prepare_api_bank_controlled.py \\
  --output-dir data/api_bank_controlled_reference_independent_v1

python - <<'PY'
import json
from pathlib import Path

p = Path("data/api_bank_controlled_reference_independent_v1/manifest.json")
m = json.loads(p.read_text())
assert m["available_tools_contract"]["all_runtime_tool_sets_reference_independent"] is True
assert m["rejected"]["public_tool_coverage"] == 0
print("VISIBLE_CURRICULUM_HOTFIX_OK", {{
    "train": m["train"],
    "dev": m["dev"],
    "train_hash": m["split_hashes"]["train"],
}})
PY

chmod +x scripts/cloud/run_v2_visible_curriculum_smoke_2gpu.sh
tmux kill-session -t v2_curriculum_smoke 2>/dev/null || true

tmux new-session -d -s v2_curriculum_smoke \\
  "cd /data/hxy/projects/RescueCredit && CUDA_VISIBLE_DEVICES=1,3 bash scripts/cloud/run_v2_visible_curriculum_smoke_2gpu.sh 2>&1 | tee outputs/v2_curriculum_smoke_driver.log"

sleep 10
tail -n 60 outputs/v2_curriculum_smoke_driver.log
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"bundle_sha256={digest}")
    print(f"paste_file_bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
