#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "scripts/analyze_route_a_bounded_recovery.py"
    payload = source.read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    output = ROOT / "dist" / "PASTE_APPWORLD_CONFIRM_RECOVERY_ANALYSIS.sh"
    output.write_text(
        f'''cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

echo "{encoded}" | base64 -d > scripts/analyze_route_a_bounded_recovery.py
echo "{digest}  scripts/analyze_route_a_bounded_recovery.py" | sha256sum -c -

PY=/data/hxy/projects/RescueCredit/.venv/bin/python
"$PY" -m py_compile scripts/analyze_route_a_bounded_recovery.py
"$PY" scripts/analyze_route_a_bounded_recovery.py \
  --root outputs/route_a_appworld_bounded_confirm_43_44_45 \
  --output outputs/route_a_appworld_bounded_confirm_43_44_45/posthoc_sensitivity.json \
  2>&1 | tee outputs/route_a_appworld_bounded_confirm_43_44_45/posthoc_sensitivity.log

echo ANALYSIS_FINISHED
echo "Result: outputs/route_a_appworld_bounded_confirm_43_44_45/posthoc_sensitivity.json"
''',
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(f"payload_sha256={digest}")


if __name__ == "__main__":
    main()

