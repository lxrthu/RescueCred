#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash, matched_epoch_order


CONFIG = {
    "seed": 42,
    "epochs": 3,
    "learning_rate": 3e-6,
    "gradient_accumulation": 8,
    "max_length": 2048,
    "beta": 1.0,
    "lora_r": 16,
    "lora_alpha": 32,
    "fp32": True,
    "preference_weight": "unit_direction",
    "sampling": "identical_natural_event_order",
}

GATE_THRESHOLDS = {
    "min_train_events": 30,
    "min_train_reverse_events": 3,
    "min_eval_events": 20,
    "min_eval_reverse_events": 2,
    "min_selection_disagreements": 3,
    "min_causal_accuracy_improvement": 0.05,
    "require_v4_wins_over_losses": True,
    "require_terminal_noninferiority": True,
    "require_progress_noninferiority": True,
}

SOURCE_PATHS = (
    "rescuecredit/toolsandbox_preference.py",
    "rescuecredit/frozen_bank.py",
    "rescuecredit/toolsandbox_credit.py",
    "environments/toolsandbox/adapter.py",
    "scripts/train_route_a_preference.py",
    "scripts/prepare_toolsandbox_v41_preference_data.py",
    "scripts/train_toolsandbox_v41_preference.py",
    "scripts/evaluate_toolsandbox_v41_preference.py",
    "scripts/freeze_toolsandbox_v41_preference_protocol.py",
    "scripts/check_toolsandbox_v41_preference_gate.py",
    "scripts/cloud/run_toolsandbox_v41_preference_seed42.sh",
    "refine-logs/TOOLSANDBOX_V41_PREFERENCE_PLAN.md",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-audit-root", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def load(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    train_path = args.data_dir / "train.jsonl"
    manifest_path = args.data_dir / "manifest.json"
    source_summary_path = args.source_audit_root / "audit_summary.json"
    source_gate_path = args.source_audit_root / "quality_gate.json"
    manifest = load(manifest_path)
    source_summary = load(source_summary_path)
    source_gate = load(source_gate_path)
    evaluation_protocol = load(args.evaluation_protocol)
    rows = read_jsonl(train_path)
    decisions = Counter(str(row["decision"]) for row in rows)
    expected_sequence = [
        str(row["event_id"])
        for epoch in range(CONFIG["epochs"])
        for row in matched_epoch_order(rows, CONFIG["seed"], epoch)
    ]
    import hashlib

    expected_sequence_sha256 = hashlib.sha256(
        "\n".join(expected_sequence).encode("utf-8")
    ).hexdigest()
    checks = {
        "source_v41_gate_passed": source_gate.get("passed") is True,
        "source_v41_protocol_validated": source_summary.get("protocol_validated")
        is True,
        "source_tool_id_interface": source_summary.get("harness_interface")
        == "tool_id_v2",
        "train_manifest_frozen": manifest.get("status") == "frozen"
        and manifest.get("role") == "train",
        "train_identity_exact": manifest.get("train_sha256")
        == file_sha256(train_path),
        "train_event_set_exact": manifest.get("event_set_hash")
        == event_set_hash(rows),
        "enough_train_events": len(rows) >= GATE_THRESHOLDS["min_train_events"],
        "enough_train_reverse_events": decisions.get("reverse_preference", 0)
        >= GATE_THRESHOLDS["min_train_reverse_events"],
        "no_protected_prompt_inputs": manifest.get("protected_outcomes_in_prompt")
        is False
        and manifest.get("official_branch_metrics_in_training_file") is False
        and manifest.get("branch_receipts_exported") is False
        and manifest.get("reference_actions_read_or_exported") is False,
        "evaluation_scenarios_frozen": evaluation_protocol.get("status")
        == "frozen_before_v4_outcomes"
        and evaluation_protocol.get("harness_interface") == "tool_id_v2"
        and evaluation_protocol.get("scenario_identity", {}).get("fresh_count")
        == 40
        and evaluation_protocol.get("scenario_identity", {}).get(
            "fresh_vs_excluded_intersection"
        )
        == [],
    }
    if not all(checks.values()):
        raise RuntimeError(f"ToolSandbox V4.1 preference preflight failed: {checks}")
    missing_sources = [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"missing frozen source paths: {missing_sources}")
    protocol = {
        "status": "frozen_before_toolsandbox_v41_preference_outcomes",
        "stage": "toolsandbox_v41_same_data_preference_seed42",
        "methods": ["mask", "v4"],
        "checks": checks,
        "config": CONFIG,
        "gate_thresholds": GATE_THRESHOLDS,
        "train_events": len(rows),
        "train_decisions": dict(sorted(decisions.items())),
        "train_sha256": file_sha256(train_path),
        "train_manifest_sha256": file_sha256(manifest_path),
        "train_event_set_hash": event_set_hash(rows),
        "source_audit_summary_sha256": file_sha256(source_summary_path),
        "source_audit_gate_sha256": file_sha256(source_gate_path),
        "source_audit_protocol_sha256": source_summary["protocol_lock_sha256"],
        "evaluation_protocol_sha256": file_sha256(args.evaluation_protocol),
        "evaluation_scenario_identity": evaluation_protocol["scenario_identity"],
        "base_model_sha256": directory_sha256(args.model),
        "source_sha256": {
            path: file_sha256(Path(path)) for path in SOURCE_PATHS
        },
        "expected_presented_event_sequence_sha256": expected_sequence_sha256,
        "reference_boundary": (
            "training labels use frozen V4 credit; model prompts contain only "
            "visible history, public schemas, and candidates; held-out official "
            "outcomes join only after adapter scoring"
        ),
        "scope": "controlled-state ToolSandbox preference learning; not autonomous task success",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
