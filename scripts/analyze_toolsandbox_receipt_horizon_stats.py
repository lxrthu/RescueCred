#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.paired_task_statistics import paired_task_analysis
from scripts.freeze_toolsandbox_receipt_horizon_stats import CONFIG, PROTOCOL_STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--v7-oof", type=Path, required=True)
    parser.add_argument("--v9-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("config") != CONFIG:
        raise ValueError("invalid receipt-horizon statistics protocol")
    if file_sha256(args.v7_oof) != protocol["artifact_sha256"]["v7_oof"]:
        raise ValueError("V7 OOF identity mismatch")
    if file_sha256(args.v9_oof) != protocol["artifact_sha256"]["v9_oof"]:
        raise ValueError("V9 OOF identity mismatch")
    if not all(
        Path(path).is_file() and file_sha256(Path(path)) == digest
        for path, digest in protocol.get("source_sha256", {}).items()
    ):
        raise ValueError("statistical audit source identity changed")
    result = paired_task_analysis(
        read_jsonl(args.v7_oof),
        read_jsonl(args.v9_oof),
        bootstrap_replicates=CONFIG["bootstrap_replicates"],
        permutation_replicates=CONFIG["permutation_replicates"],
        seed=CONFIG["seed"],
        alpha=CONFIG["alpha"],
    )
    if result["events"] != 126:
        raise ValueError("statistical audit requires exactly 126 paired events")
    claim_boundary = (
        "The paired frozen-task audit supports the original positive routing gates."
        if result["positive_routing_claim_supported"]
        else "The tested two-step offline receipt representation does not support a positive routing claim. Statistical classification is limited to the paired frozen ToolSandbox tasks."
    )
    payload = {
        "status": "completed",
        "stage": "toolsandbox_receipt_horizon_statistical_audit_seed42",
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "v7_oof_sha256": file_sha256(args.v7_oof),
        "v9_oof_sha256": file_sha256(args.v9_oof),
        **result,
        "claim_boundary": claim_boundary,
        "model_selection_reopened": False,
        "uncertainty_scope": protocol["uncertainty_scope"],
        "next_step": "freeze V6/V7/V9 as a negative boundary; do not extend receipt horizon on these events",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
