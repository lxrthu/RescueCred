#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.train_toolsandbox_v7_active_shadow as v7
from scripts.freeze_toolsandbox_v9_protocol import CONFIG, PROTOCOL_STATUS


def main() -> None:
    v7.CONFIG = CONFIG
    v7.PROTOCOL_STATUS = PROTOCOL_STATUS
    v7.main()
    output_index = sys.argv.index("--output-dir") + 1
    summary_path = Path(sys.argv[output_index]) / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["stage"] = "toolsandbox_v9_two_step_nested_cross_task_oof"
    summary["feature_variant"] = "first_two_branch_receipts"
    summary["probe_horizon"] = 2
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
