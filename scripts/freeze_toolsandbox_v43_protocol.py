#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash
from scripts.freeze_toolsandbox_v42_protocol import (
    CONFIRMATION_THRESHOLDS as V42_CONFIRMATION_THRESHOLDS,
    DEVELOPMENT_THRESHOLDS as V42_DEVELOPMENT_THRESHOLDS,
)
from scripts.prepare_toolsandbox_v43_training_data import THRESHOLDS as DATA_THRESHOLDS
from scripts.train_route_a_preference import balanced_causal_epoch_order
from scripts.train_toolsandbox_v43_preference import PROTOCOL_STATUS


CONFIG = {
    "seed": 42,
    "epochs": 3,
    "learning_rate": 3e-6,
    "gradient_accumulation": 8,
    "max_length": 2048,
    "beta": 1.0,
    "absolute_margin_coef": 1.0,
    "target_margin": 0.05,
    "reference_anchor_coef": 0.25,
    "presentations_per_epoch": 60,
    "lora_r": 16,
    "lora_alpha": 32,
    "fp32": True,
    "preference_weight": "unit_direction",
    "sampling": "identical_multi_prefix_class_balanced",
}

DEVELOPMENT_THRESHOLDS = {
    **V42_DEVELOPMENT_THRESHOLDS,
    "min_class_conditional_shift_gap": 0.02,
}
CONFIRMATION_THRESHOLDS = {
    **V42_CONFIRMATION_THRESHOLDS,
    "min_class_conditional_shift_gap": 0.02,
}

SOURCE_PATHS = (
    "environments/toolsandbox/adapter.py",
    "rescuecredit/toolsandbox_preference.py",
    "rescuecredit/toolsandbox_credit.py",
    "scripts/audit_toolsandbox_signal.py",
    "scripts/freeze_toolsandbox_v4_protocol.py",
    "scripts/prepare_toolsandbox_v43_training_data.py",
    "scripts/train_route_a_preference.py",
    "scripts/train_toolsandbox_v43_preference.py",
    "scripts/evaluate_toolsandbox_v43_preference.py",
    "scripts/freeze_toolsandbox_v43_protocol.py",
    "scripts/check_toolsandbox_v43_gate.py",
    "scripts/cloud/run_toolsandbox_v43_seed42.sh",
    "refine-logs/TOOLSANDBOX_V43_PLAN.md",
    "refine-logs/TOOLSANDBOX_V43_PLAN_20260720.md",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mining-audit-root", type=Path, required=True)
    parser.add_argument("--mining-protocol", type=Path, required=True)
    parser.add_argument("--old-training-protocol", type=Path, required=True)
    parser.add_argument("--development-data-dir", type=Path, required=True)
    parser.add_argument("--development-audit-root", type=Path, required=True)
    parser.add_argument("--v42-development-gate", type=Path, required=True)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--old-v42-confirmation-protocol", type=Path, required=True)
    parser.add_argument("--v42-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_path = args.data_dir / "train.jsonl"
    manifest_path = args.data_dir / "manifest.json"
    data_gate_path = args.data_dir / "data_gate.json"
    mining_summary_path = args.mining_audit_root / "audit_summary.json"
    mining_gate_path = args.mining_audit_root / "quality_gate.json"
    dev_manifest_path = args.development_data_dir / "manifest.json"
    dev_public_path = args.development_data_dir / "events.public.jsonl"
    dev_private_path = args.development_data_dir / "outcomes.private.jsonl"
    dev_summary_path = args.development_audit_root / "audit_summary.json"
    dev_gate_path = args.development_audit_root / "quality_gate.json"

    manifest = _load(manifest_path)
    data_gate = _load(data_gate_path)
    mining_summary = _load(mining_summary_path)
    mining_gate = _load(mining_gate_path)
    mining_protocol = _load(args.mining_protocol)
    old_training = _load(args.old_training_protocol)
    dev_manifest = _load(dev_manifest_path)
    dev_summary = _load(dev_summary_path)
    dev_audit_gate = _load(dev_gate_path)
    old_v42_dev_gate = _load(args.v42_development_gate)
    confirmation = _load(args.confirmation_protocol)
    old_confirmation = _load(args.old_v42_confirmation_protocol)
    rows = read_jsonl(train_path)
    decisions = Counter(str(row["decision"]) for row in rows)

    sequence_rows = [
        row
        for epoch in range(CONFIG["epochs"])
        for row in balanced_causal_epoch_order(
            rows,
            CONFIG["seed"],
            epoch,
            CONFIG["presentations_per_epoch"],
        )
    ]
    sequence_decisions = Counter(str(row["decision"]) for row in sequence_rows)
    sequence_sha256 = hashlib.sha256(
        "\n".join(str(row["event_id"]) for row in sequence_rows).encode("utf-8")
    ).hexdigest()
    expected_labels = {
        "mask": {"b_over_a": len(sequence_rows)},
        "v43": {
            "a_over_b": sequence_decisions["reverse_preference"],
            "b_over_a": sequence_decisions["rescue_preference"],
        },
    }
    mining_identity = mining_protocol.get("scenario_identity", {})
    old_training_identity = old_training.get("scenario_identity", {})
    confirm_identity = confirmation.get("scenario_identity", {})
    old_confirm_identity = old_confirmation.get("scenario_identity", {})
    old_confirm_outcome = (
        args.v42_root / "fresh_confirm_offset165_h8" / "audit_summary.json"
    )
    checks = {
        "data_gate_passed": data_gate.get("passed") is True
        and manifest.get("passed") is True
        and manifest.get("status") == "frozen",
        "data_thresholds_frozen": data_gate.get("thresholds") == DATA_THRESHOLDS,
        "train_identity_exact": manifest.get("train_sha256")
        == file_sha256(train_path)
        and manifest.get("event_set_hash") == event_set_hash(rows),
        "train_has_diverse_reverse": decisions.get("reverse_preference", 0)
        >= DATA_THRESHOLDS["min_reverse_events"]
        and manifest.get("reverse_tasks", 0) >= DATA_THRESHOLDS["min_reverse_tasks"],
        "training_reference_boundary": manifest.get(
            "official_branch_metrics_in_training_file"
        )
        is False
        and manifest.get("protected_outcomes_in_prompt") is False
        and manifest.get("branch_receipts_exported") is False
        and manifest.get("reference_actions_read_or_exported") is False,
        "mining_audit_bound": mining_summary.get("protocol_lock_sha256")
        == file_sha256(args.mining_protocol)
        and mining_summary.get("max_events_per_scenario") == 4
        and mining_summary.get("protocol_validated") is True
        and mining_gate.get("mechanism_passed") is True,
        "mining_reuses_training_tasks_only": mining_protocol.get("scenario_offset")
        == old_training.get("scenario_offset")
        == 85
        and mining_identity.get("fresh_hashes")
        == old_training_identity.get("fresh_hashes"),
        "development_is_known_offset125": dev_summary.get("scenario_offset") == 125
        and dev_summary.get("protocol_validated") is True
        and dev_audit_gate.get("mechanism_passed") is True
        and old_v42_dev_gate.get("passed") is False,
        "development_data_bound": dev_manifest.get("role") == "evaluation"
        and dev_manifest.get("public_sha256") == file_sha256(dev_public_path)
        and dev_manifest.get("private_sha256") == file_sha256(dev_private_path),
        "confirmation_reproduces_unseen_offset165": confirmation.get(
            "scenario_offset"
        )
        == old_confirmation.get("scenario_offset")
        == 165
        and confirm_identity.get("fresh_hashes")
        == old_confirm_identity.get("fresh_hashes")
        and confirm_identity.get("fresh_count") == 40,
        "old_confirmation_outcomes_absent": not old_confirm_outcome.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"ToolSandbox V4.3 protocol preflight failed: {checks}")
    missing_sources = [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"missing frozen source paths: {missing_sources}")

    protocol = {
        "status": PROTOCOL_STATUS,
        "stage": "toolsandbox_v43_multi_prefix_anchored_seed42",
        "methods": ["mask", "v43"],
        "checks": checks,
        "config": CONFIG,
        "data_thresholds": DATA_THRESHOLDS,
        "gate_thresholds": {
            "development": DEVELOPMENT_THRESHOLDS,
            "confirmation": CONFIRMATION_THRESHOLDS,
        },
        "train_events": len(rows),
        "train_decisions": dict(sorted(decisions.items())),
        "train_sha256": file_sha256(train_path),
        "train_manifest_sha256": file_sha256(manifest_path),
        "train_data_gate_sha256": file_sha256(data_gate_path),
        "train_event_set_hash": event_set_hash(rows),
        "mining": {
            "protocol_sha256": file_sha256(args.mining_protocol),
            "old_training_protocol_sha256": file_sha256(
                args.old_training_protocol
            ),
            "audit_summary_sha256": file_sha256(mining_summary_path),
            "audit_gate_sha256": file_sha256(mining_gate_path),
            "scenario_identity": mining_identity,
        },
        "development": {
            "known_before_v43_training": True,
            "event_set_hash": dev_manifest["event_set_hash"],
            "events": dev_manifest["events"],
            "manifest_sha256": file_sha256(dev_manifest_path),
            "public_sha256": file_sha256(dev_public_path),
            "private_sha256": file_sha256(dev_private_path),
            "audit_summary_sha256": file_sha256(dev_summary_path),
            "audit_gate_sha256": file_sha256(dev_gate_path),
            "audit_protocol_sha256": dev_summary["protocol_lock_sha256"],
            "scenario_hashes": dev_summary["selected_scenario_hashes"],
            "v42_gate_sha256": file_sha256(args.v42_development_gate),
        },
        "confirmation": {
            "outcomes_unobserved_at_freeze": True,
            "protocol_sha256": file_sha256(args.confirmation_protocol),
            "old_v42_protocol_sha256": file_sha256(
                args.old_v42_confirmation_protocol
            ),
            "scenario_identity": confirm_identity,
        },
        "base_model_sha256": directory_sha256(args.model),
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "expected_presented_event_sequence_sha256": sequence_sha256,
        "expected_presented_source_decisions": dict(
            sorted(sequence_decisions.items())
        ),
        "expected_presented_decisions": expected_labels,
        "reference_boundary": (
            "multi-prefix training uses only visible histories, public schemas, "
            "candidate actions, and frozen V4 direction; official outcomes join "
            "only after scoring"
        ),
        "scope": "controlled-state ToolSandbox preference learning; not autonomous task success",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
