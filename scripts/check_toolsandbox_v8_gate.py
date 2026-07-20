#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.check_toolsandbox_v7_gate as v7
from scripts.freeze_toolsandbox_v8_protocol import GATE, PROTOCOL_STATUS


def main() -> None:
    v7.GATE = GATE
    v7.PROTOCOL_STATUS = PROTOCOL_STATUS
    try:
        v7.main()
    except SystemExit as error:
        output_index = sys.argv.index("--output") + 1
        output = Path(sys.argv[output_index])
        if output.is_file():
            gate = json.loads(output.read_text(encoding="utf-8"))
            gate["stage"] = "toolsandbox_v8_visible_state_feasibility_gate_seed42"
            gate["feature_variant"] = "explicit_one_step_visible_state"
            output.write_text(
                json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(gate, ensure_ascii=False, indent=2))
        raise error


if __name__ == "__main__":
    main()
