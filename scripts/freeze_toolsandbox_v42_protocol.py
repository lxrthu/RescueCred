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
from scripts.train_route_a_preference import balanced_causal_epoch_order
from scripts.train_toolsandbox_v42_preference import PROTOCOL_STATUS


CONFIG = {
    "seed": 42,
    "epochs": 3,
    "learning_rate": 3e-6,
    "gradient_accumulation": 8,
    "max_length": 2048,
    "beta": 1.0,
    "absolute_margin_coef": 1.0,
    "target_margin": 0.05,
    "presentations_per_epoch": 36,
    "lora_r": 16,
    "lora_alpha": 32,
    "fp32": True,
    "preference_weight": "unit_direction",
    "sampling": "identical_class_balanced_rescue_reverse",
}

DEVELOPMENT_THRESHOLDS = {
    "min_eval_events": 20,
    "min_eval_reverse_events": 2,
    "min_selection_disagreements": 1,
    "require_positive_causal_accuracy_improvement": True,
    "require_v42_wins_over_losses": True,
    "require_terminal_noninferiority": True,
    "require_progress_noninferiority": True,
}

CONFIRMATION_THRESHOLDS = {
    "min_eval_events": 20,
    "min_eval_reverse_events": 2,
    "min_selection_disagreements": 3,
    "min_causal_accuracy_improvement": 0.05,
    "require_v42_wins_over_losses": True,
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
    "scripts/train_toolsandbox_v42_preference.py",
    "scripts/evaluate_toolsandbox_v42_preference.py",
    "scripts/freeze_toolsandbox_v42_protocol.py",
    "scripts/check_toolsandbox_v42_gate.py",
    "scripts/cloud/run_toolsandbox_v42_seed42.sh",
    "refine-logs/TOOLSANDBOX_V42_PLAN.md",
    "refine-logs/TOOLSANDBOX_V42_PLAN_20260720.md",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-audit-root", type=Path, required=True)
    parser.add_argument("--development-data-dir", type=Path, required=True)
    parser.add_argument("--development-audit-root", type=Path, required=True)
    parser.add_argument("--v41-development-gate", type=Path, required=True)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_path = args.data_dir / "train.jsonl"
    train_manifest_path = args.data_dir / "manifest.json"
    source_summary_path = args.source_audit_root / "audit_summary.json"
    source_gate_path = args.source_audit_root / "quality_gate.json"
    dev_manifest_path = args.development_data_dir / "manifest.json"
    dev_public_path = args.development_data_dir / "events.public.jsonl"
    dev_private_path = args.development_data_dir / "outcomes.private.jsonl"
    dev_summary_path = args.development_audit_root / "audit_summary.json"
    dev_gate_path = args.development_audit_root / "quality_gate.json"

    train_manifest = _load(train_manifest_path)
    source_summary = _load(source_summary_path)
    source_gate = _load(source_gate_path)
    dev_manifest = _load(dev_manifest_path)
    dev_summary = _load(dev_summary_path)
    dev_audit_gate = _load(dev_gate_path)
    old_v41_gate = _load(args.v41_development_gate)
    confirmation = _load(args.confirmation_protocol)
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
    sequence_ids = [str(row["event_id"]) for row in sequence_rows]
    sequence_decisions = Counter(str(row["decision"]) for row in sequence_rows)
    sequence_sha256 = hashlib.sha256(
        "\n".join(sequence_ids).encode("utf-8")
    ).hexdigest()
    expected_labels = {
        "mask": {"b_over_a": len(sequence_rows)},
        "v42": {
            "a_over_b": sequence_decisions["reverse_preference"],
            "b_over_a": sequence_decisions["rescue_preference"],
        },
    }

    confirmation_identity = confirmation.get("scenario_identity", {})
    checks = {
        "source_v41_gate_passed": source_gate.get("passed") is True,
        "source_v41_protocol_validated": source_summary.get("protocol_validated")
        is True,
        "source_tool_id_interface": source_summary.get("harness_interface")
        == "tool_id_v2",
        "train_manifest_frozen": train_manifest.get("status") == "frozen"
        and train_manifest.get("role") == "train",
        "train_identity_exact": train_manifest.get("train_sha256")
        == file_sha256(train_path),
        "train_event_set_exact": train_manifest.get("event_set_hash")
        == event_set_hash(rows),
        "exact_train_events": len(rows) == CONFIG["presentations_per_epoch"] == 36,
        "both_training_classes_present": decisions.get("rescue_preference", 0) > 0
        and decisions.get("reverse_preference", 0) >= 3,
        "balanced_presentations": sequence_decisions
        == Counter({"rescue_preference": 54, "reverse_preference": 54}),
        "no_protected_training_inputs": train_manifest.get(
            "protected_outcomes_in_prompt"
        )
        is False
        and train_manifest.get("official_branch_metrics_in_training_file") is False
        and train_manifest.get("branch_receipts_exported") is False
        and train_manifest.get("reference_actions_read_or_exported") is False,
        "development_is_old_offset125": dev_summary.get("scenario_offset") == 125
        and dev_summary.get("harness_interface") == "tool_id_v2"
        and dev_summary.get("protocol_validated") is True,
        "development_data_bound": dev_manifest.get("role") == "evaluation"
        and dev_manifest.get("public_sha256") == file_sha256(dev_public_path)
        and dev_manifest.get("private_sha256") == file_sha256(dev_private_path)
        and dev_manifest.get("source_summary_sha256") == file_sha256(dev_summary_path)
        and dev_manifest.get("source_gate_sha256") == file_sha256(dev_gate_path),
        "development_mechanism_passed": dev_audit_gate.get("mechanism_passed")
        is True,
        "v41_development_was_negative": old_v41_gate.get("passed") is False
        and old_v41_gate.get("stage")
        == "toolsandbox_v41_same_data_preference_seed42_gate",
        "confirmation_frozen_offset165": confirmation.get("status")
        == "frozen_before_v4_outcomes"
        and confirmation.get("scenario_offset") == 165
        and confirmation.get("limit") == 40
        and confirmation.get("horizon") == 8
        and confirmation.get("event_search_steps") == 8
        and confirmation.get("harness_interface") == "tool_id_v2",
        "confirmation_exact_and_disjoint": confirmation_identity.get("fresh_count")
        == 40
        and confirmation_identity.get("fresh_vs_excluded_intersection") == []
        and len(confirmation_identity.get("excluded_protocols", [])) == 4,
    }
    if not all(checks.values()):
        raise RuntimeError(f"ToolSandbox V4.2 protocol preflight failed: {checks}")
    missing_sources = [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"missing frozen source paths: {missing_sources}")

    protocol = {
        "status": PROTOCOL_STATUS,
        "stage": "toolsandbox_v42_balanced_margin_seed42",
        "methods": ["mask", "v42"],
        "checks": checks,
        "config": CONFIG,
        "gate_thresholds": {
            "development": DEVELOPMENT_THRESHOLDS,
            "confirmation": CONFIRMATION_THRESHOLDS,
        },
        "train_events": len(rows),
        "train_decisions": dict(sorted(decisions.items())),
        "train_sha256": file_sha256(train_path),
        "train_manifest_sha256": file_sha256(train_manifest_path),
        "train_event_set_hash": event_set_hash(rows),
        "source_audit_summary_sha256": file_sha256(source_summary_path),
        "source_audit_gate_sha256": file_sha256(source_gate_path),
        "source_audit_protocol_sha256": source_summary["protocol_lock_sha256"],
        "development": {
            "known_before_v42_training": True,
            "event_set_hash": dev_manifest["event_set_hash"],
            "events": dev_manifest["events"],
            "manifest_sha256": file_sha256(dev_manifest_path),
            "public_sha256": file_sha256(dev_public_path),
            "private_sha256": file_sha256(dev_private_path),
            "audit_summary_sha256": file_sha256(dev_summary_path),
            "audit_gate_sha256": file_sha256(dev_gate_path),
            "audit_protocol_sha256": dev_summary["protocol_lock_sha256"],
            "scenario_hashes": dev_summary["selected_scenario_hashes"],
            "v41_gate_sha256": file_sha256(args.v41_development_gate),
        },
        "confirmation": {
            "outcomes_unobserved_at_freeze": True,
            "protocol_sha256": file_sha256(args.confirmation_protocol),
            "scenario_identity": confirmation_identity,
        },
        "base_model_sha256": directory_sha256(args.model),
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "expected_presented_event_sequence_sha256": sequence_sha256,
        "expected_presented_source_decisions": dict(
            sorted(sequence_decisions.items())
        ),
        "expected_presented_decisions": expected_labels,
        "reference_boundary": (
            "training uses frozen V4 causal directions; prompts contain only visible "
            "history, public schemas, and candidate actions; official branch outcomes "
            "join only after adapter scoring"
        ),
        "scope": "controlled-state ToolSandbox preference learning; not autonomous task success",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
