#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash
from scripts.train_route_a_preference import balanced_causal_epoch_order
from scripts.train_toolsandbox_v43_preference import V45_PROTOCOL_STATUS


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
    "presentations_per_epoch": 126,
    "lora_r": 16,
    "lora_alpha": 32,
    "fp32": True,
    "preference_weight": "unit_direction",
    "sampling": "identical_candidate_diversity_class_balanced",
}

DEVELOPMENT_THRESHOLDS = {
    "min_eval_events": 40,
    "min_eval_rescue_events": 5,
    "min_eval_reverse_events": 5,
    "min_selection_disagreements": 3,
    "min_class_conditional_shift_gap": 0.02,
    "require_positive_causal_accuracy_improvement": True,
    "require_v45_wins_over_losses": True,
    "require_terminal_noninferiority": True,
    "require_progress_noninferiority": True,
}
CONFIRMATION_THRESHOLDS = {
    **DEVELOPMENT_THRESHOLDS,
    "min_causal_accuracy_improvement": 0.05,
}

SOURCE_PATHS = (
    "environments/toolsandbox/adapter.py",
    "rescuecredit/frozen_bank.py",
    "rescuecredit/logging.py",
    "rescuecredit/toolsandbox_preference.py",
    "scripts/train_route_a_preference.py",
    "scripts/train_toolsandbox_v43_preference.py",
    "scripts/evaluate_toolsandbox_v43_preference.py",
    "scripts/freeze_toolsandbox_v45_candidate_protocol.py",
    "scripts/freeze_toolsandbox_v45_learner_protocol.py",
    "scripts/check_toolsandbox_v45_gate.py",
    "scripts/cloud/run_toolsandbox_v45_seed42.sh",
    "refine-logs/TOOLSANDBOX_V45_PLAN.md",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-audit-root", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--development-protocol", type=Path, required=True)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace frozen protocol: {args.output}")

    train_path = args.data_dir / "train.jsonl"
    manifest_path = args.data_dir / "manifest.json"
    data_gate_path = args.data_dir / "data_gate.json"
    summary_path = args.source_audit_root / "audit_summary.json"
    required = [train_path, manifest_path, data_gate_path, summary_path, args.source_protocol,
                args.development_protocol, args.confirmation_protocol]
    missing = [str(path) for path in required if not path.is_file()]
    missing += [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing V4.5 learner inputs: {missing}")

    rows = read_jsonl(train_path)
    manifest, data_gate, summary = _load(manifest_path), _load(data_gate_path), _load(summary_path)
    source_protocol = _load(args.source_protocol)
    development, confirmation = _load(args.development_protocol), _load(args.confirmation_protocol)
    decisions = Counter(str(row["decision"]) for row in rows)
    sequence = [row for epoch in range(CONFIG["epochs"]) for row in balanced_causal_epoch_order(rows, 42, epoch, 126)]
    sequence_decisions = Counter(str(row["decision"]) for row in sequence)
    sequence_hash = hashlib.sha256("\n".join(str(row["event_id"]) for row in sequence).encode()).hexdigest()
    expected_labels = {
        "mask": {"b_over_a": len(sequence)},
        "v45": {"a_over_b": sequence_decisions["reverse_preference"], "b_over_a": sequence_decisions["rescue_preference"]},
    }
    dev_hashes = development.get("scenario_identity", {}).get("fresh_hashes", [])
    confirm_hashes = confirmation.get("scenario_identity", {}).get("fresh_hashes", [])
    train_hashes = source_protocol.get("scenario_identity", {}).get("fresh_hashes", [])
    checks = {
        "v44_data_gate_passed": data_gate.get("passed") is True and manifest.get("status") == "frozen",
        "exact_train_identity": manifest.get("train_sha256") == file_sha256(train_path) and manifest.get("event_set_hash") == event_set_hash(rows),
        "exact_train_counts": len(rows) == 126 and decisions == Counter({"rescue_preference": 41, "reverse_preference": 85}),
        "source_audit_bound": summary.get("protocol_validated") is True and summary.get("protocol_lock_sha256") == file_sha256(args.source_protocol),
        "no_protected_training_inputs": manifest.get("official_branch_metrics_in_training_file") is False and manifest.get("protected_outcomes_in_prompt") is False and manifest.get("branch_receipts_exported") is False and manifest.get("reference_actions_read_or_exported") is False,
        "evaluation_protocols_frozen_preoutcome": development.get("evaluation_role") == "development" and confirmation.get("evaluation_role") == "confirmation",
        "evaluation_scenarios_disjoint": len(dev_hashes) == len(confirm_hashes) == 40 and not (set(train_hashes) & set(dev_hashes)) and not (set(train_hashes) & set(confirm_hashes)) and not (set(dev_hashes) & set(confirm_hashes)),
        "balanced_matched_presentations": sequence_decisions == Counter({"rescue_preference": 189, "reverse_preference": 189}),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V4.5 learner preflight failed: {checks}")

    protocol = {
        "status": V45_PROTOCOL_STATUS,
        "stage": "toolsandbox_v45_matched_anchored_candidate_diversity_seed42",
        "methods": ["mask", "v45"],
        "checks": checks,
        "config": CONFIG,
        "gate_thresholds": {"development": DEVELOPMENT_THRESHOLDS, "confirmation": CONFIRMATION_THRESHOLDS},
        "train_events": len(rows),
        "train_decisions": dict(sorted(decisions.items())),
        "train_sha256": file_sha256(train_path),
        "train_manifest_sha256": file_sha256(manifest_path),
        "train_data_gate_sha256": file_sha256(data_gate_path),
        "train_event_set_hash": event_set_hash(rows),
        "source_audit_summary_sha256": file_sha256(summary_path),
        "source_audit_protocol_sha256": file_sha256(args.source_protocol),
        "development": {"outcomes_unobserved_at_freeze": True, "protocol_sha256": file_sha256(args.development_protocol), "scenario_identity": development["scenario_identity"]},
        "confirmation": {"outcomes_unobserved_at_freeze": True, "protocol_sha256": file_sha256(args.confirmation_protocol), "scenario_identity": confirmation["scenario_identity"]},
        "base_model_sha256": directory_sha256(args.model),
        "source_sha256": {path: file_sha256(Path(path)) for path in SOURCE_PATHS},
        "expected_presented_event_sequence_sha256": sequence_hash,
        "expected_presented_source_decisions": dict(sorted(sequence_decisions.items())),
        "expected_presented_decisions": expected_labels,
        "reference_boundary": "training prompts contain only visible history, public schemas, and frozen candidates; official branch outcomes are used only for frozen direction labels and offline evaluation",
        "scope": "controlled-state ToolSandbox same-distribution preference learning; not autonomous task success",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
