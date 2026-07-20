#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import scripts.train_toolsandbox_v7_active_shadow as v7
from scripts.freeze_toolsandbox_v8_protocol import CONFIG, PROTOCOL_STATUS


def main() -> None:
    # Reuse the reviewed V7 nested task cross-fit unchanged; only the frozen
    # protocol identity and feature cache differ in V8.
    v7.CONFIG = CONFIG
    v7.PROTOCOL_STATUS = PROTOCOL_STATUS
    v7.main()

    # v7.main parses the shared CLI and writes into its --output-dir. Recover
    # that path without duplicating its parser solely to version the stage name.
    import sys

    output_index = sys.argv.index("--output-dir") + 1
    summary_path = Path(sys.argv[output_index]) / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["stage"] = "toolsandbox_v8_visible_state_nested_cross_task_oof"
    summary["feature_variant"] = "explicit_one_step_visible_state"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
