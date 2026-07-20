#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import summarize_evaluation_rows
from scripts.freeze_toolsandbox_v46_protocol import THRESHOLDS
from scripts.train_toolsandbox_v43_preference import V46_PROTOCOL_STATUS

TOL = 1e-12


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summary_exact(summary: dict, rows: list[dict]) -> bool:
    expected = summarize_evaluation_rows(rows)
    for key, value in expected.items():
        actual = summary.get(key)
        if isinstance(value, float):
            if actual is None or abs(float(actual) - value) > TOL:
                return False
        elif actual != value:
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    for name in (
        "mask-eval",
        "control-eval",
        "v46-eval",
        "control-run",
        "v46-run",
        "mask-results",
        "control-results",
        "v46-results",
        "protocol-lock",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    mask, control, v46 = load(a.mask_eval), load(a.control_eval), load(a.v46_eval)
    cr, vr, protocol = load(a.control_run), load(a.v46_run), load(a.protocol_lock)
    mr, ctr, vrr = (
        read_jsonl(a.mask_results),
        read_jsonl(a.control_results),
        read_jsonl(a.v46_results),
    )
    mb, cb, vb = ({str(r["event_id"]): r for r in rows} for rows in (mr, ctr, vrr))
    same = set(mb) == set(cb) == set(vb) and len(mb) == len(mr) == len(ctr) == len(vrr)
    disagreements = [
        i for i in sorted(mb) if same and mb[i]["selected"] != vb[i]["selected"]
    ]
    wins = sum(
        vb[i]["causal_correct"] and not mb[i]["causal_correct"] for i in disagreements
    )
    losses = sum(
        mb[i]["causal_correct"] and not vb[i]["causal_correct"] for i in disagreements
    )
    shifts = {"rescue_preference": [], "reverse_preference": []}
    for i in sorted(mb):
        d = mb[i]["decision"]
        shifts[d].append(
            float(vb[i]["margin_b_over_a"]) - float(mb[i]["margin_b_over_a"])
        )

    def mean(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    rescue_shift, reverse_shift = (
        mean(shifts["rescue_preference"]),
        mean(shifts["reverse_preference"]),
    )
    expected = protocol["train_events"] * protocol["config"]["epochs"]
    development_manifest_path = (
        Path(protocol["development"]["data_dir"]) / "manifest.json"
    )
    development_manifest = load(development_manifest_path)
    source_ok = all(
        Path(x).is_file() and file_sha256(Path(x)) == h
        for x, h in protocol.get("source_sha256", {}).items()
    )
    integrity = {
        "protocol_frozen": protocol.get("status") == V46_PROTOCOL_STATUS
        and protocol.get("thresholds") == THRESHOLDS,
        "source_identity": source_ok,
        "runs_bound": cr.get("protocol_lock_sha256")
        == vr.get("protocol_lock_sha256")
        == file_sha256(a.protocol_lock),
        "evaluation_protocol_bound": mask.get("protocol_lock_sha256")
        == control.get("protocol_lock_sha256")
        == v46.get("protocol_lock_sha256")
        == file_sha256(a.protocol_lock),
        "common_mask_start": cr.get("mask_adapter_sha256")
        == vr.get("mask_adapter_sha256")
        == protocol.get("mask_adapter_sha256"),
        "matched_budget_sequence": cr.get("active_event_presentations")
        == vr.get("active_event_presentations")
        == expected
        and cr.get("presented_event_sequence_sha256")
        == vr.get("presented_event_sequence_sha256")
        == protocol.get("expected_presented_event_sequence_sha256"),
        "results_bound": mask.get("results_sha256") == file_sha256(a.mask_results)
        and control.get("results_sha256") == file_sha256(a.control_results)
        and v46.get("results_sha256") == file_sha256(a.v46_results),
        "same_event_set": same
        and mask.get("event_set_hash")
        == control.get("event_set_hash")
        == v46.get("event_set_hash"),
        "method_roles": control.get("method") == cr.get("method") == "control"
        and v46.get("method") == vr.get("method") == "v46",
        "run_and_adapter_identity": control.get("run_summary_sha256")
        == file_sha256(a.control_run)
        and v46.get("run_summary_sha256") == file_sha256(a.v46_run)
        and control.get("adapter_sha256") == cr.get("adapter_sha256")
        and v46.get("adapter_sha256") == vr.get("adapter_sha256"),
        "evaluation_files_bound": development_manifest_path.is_file()
        and file_sha256(development_manifest_path)
        == protocol["development"]["manifest_sha256"]
        and mask.get("public_events_sha256")
        == control.get("public_events_sha256")
        == v46.get("public_events_sha256")
        == development_manifest.get("public_sha256")
        and mask.get("private_outcomes_sha256")
        == control.get("private_outcomes_sha256")
        == v46.get("private_outcomes_sha256")
        == development_manifest.get("private_sha256"),
        "raw_metrics_recomputed": summary_exact(mask, mr)
        and summary_exact(control, ctr)
        and summary_exact(v46, vrr),
        "development_only": mask.get("evaluation_role")
        == control.get("evaluation_role")
        == v46.get("evaluation_role")
        == "development",
    }
    outcomes = {
        "enough_events": int(v46["valid_events"]) >= THRESHOLDS["min_events"],
        "enough_disagreements": len(disagreements) >= THRESHOLDS["min_disagreements"],
        "overall_beats_frozen_mask": v46["causal_accuracy"]
        > mask["causal_accuracy"] + TOL,
        "overall_beats_matched_control": v46["causal_accuracy"]
        > control["causal_accuracy"] + TOL,
        "rescue_noninferiority": v46["rescue_accuracy"]
        >= mask["rescue_accuracy"] - TOL,
        "reverse_improvement": v46["reverse_accuracy"] > mask["reverse_accuracy"] + TOL,
        "signed_rescue_shift": rescue_shift >= -TOL,
        "signed_reverse_shift": reverse_shift
        <= -THRESHOLDS["min_reverse_margin_decrease"] + TOL,
        "wins_over_losses": wins > losses,
    }
    passed = all(integrity.values()) and all(outcomes.values())
    gate = {
        "passed": passed,
        "stage": "toolsandbox_v46_development_gate_seed42",
        "integrity_checks": integrity,
        "outcome_checks": outcomes,
        "thresholds": THRESHOLDS,
        "events": v46["valid_events"],
        "selection_disagreements": len(disagreements),
        "v46_wins": wins,
        "v46_losses": losses,
        "mask_accuracy": mask["causal_accuracy"],
        "control_accuracy": control["causal_accuracy"],
        "v46_accuracy": v46["causal_accuracy"],
        "v46_vs_mask": v46["causal_accuracy"] - mask["causal_accuracy"],
        "v46_vs_control": v46["causal_accuracy"] - control["causal_accuracy"],
        "mask_rescue_accuracy": mask["rescue_accuracy"],
        "v46_rescue_accuracy": v46["rescue_accuracy"],
        "mask_reverse_accuracy": mask["reverse_accuracy"],
        "v46_reverse_accuracy": v46["reverse_accuracy"],
        "mean_rescue_margin_shift": rescue_shift,
        "mean_reverse_margin_shift": reverse_shift,
        "scope": protocol["scope"],
        "next_step": "freeze a new scenario-profile confirmation"
        if passed
        else "revise or stop the residual learner",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
