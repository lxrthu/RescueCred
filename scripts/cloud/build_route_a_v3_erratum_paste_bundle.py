#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/audit_route_a_v3_expanded_gate_erratum.py"
TEST = ROOT / "tests/test_route_a_v3_expanded_erratum.py"
OUTPUT = ROOT / "dist/PASTE_APPWORLD_V3_GATE_ERRATUM_TO_SERVER.sh"


def main() -> None:
    files = {
        SOURCE.relative_to(ROOT).as_posix(): SOURCE,
        TEST.relative_to(ROOT).as_posix(): TEST,
    }
    encoded = {
        name: base64.b64encode(path.read_bytes()).decode("ascii")
        for name, path in files.items()
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    writes = "\n".join(
        f'Path({name!r}).write_bytes(base64.b64decode({payload!r}))'
        for name, payload in encoded.items()
    )
    checks = "\n".join(f'echo "{digest}  {name}" | sha256sum -c -' for name, digest in hashes.items())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        f'''(
set -e
cd /data/hxy/projects/RescueCredit
export PROMPT_COMMAND=

python - <<'PY'
import base64
from pathlib import Path
for malformed in (
    Path(r"scripts\\audit_route_a_v3_expanded_gate_erratum.py"),
    Path(r"tests\\test_route_a_v3_expanded_erratum.py"),
):
    if malformed.is_file():
        malformed.unlink()
{writes}
print("WROTE V3 gate erratum audit")
PY

{checks}

PY=/data/hxy/projects/RescueCredit/.venv/bin/python
"$PY" -m py_compile scripts/audit_route_a_v3_expanded_gate_erratum.py
"$PY" -m pytest -q tests/test_route_a_v3_expanded_erratum.py

ROOT=outputs/route_a_v3_expanded_seed42
"$PY" scripts/audit_route_a_v3_expanded_gate_erratum.py \
  --original-gate "$ROOT/gate.json" \
  --v3-run "$ROOT/v3/run_summary.json" \
  --protocol-lock "$ROOT/protocol_lock.json" \
  --output "$ROOT/gate_erratum.json"

cat "$ROOT/gate_erratum.json"
)
''',
        encoding="utf-8",
        newline="\n",
    )
    print(OUTPUT)
    print(f"paste_file_sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
