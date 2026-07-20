#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.check_toolsandbox_v7_gate as v7
from scripts.freeze_toolsandbox_v9_protocol import GATE, PROTOCOL_STATUS


def main() -> None:
    v7.GATE = GATE
    v7.PROTOCOL_STATUS = PROTOCOL_STATUS
    try:
        v7.main()
    except SystemExit:
        output_index = sys.argv.index("--output") + 1
        output = Path(sys.argv[output_index])
        protocol_index = sys.argv.index("--protocol-lock") + 1
        protocol_path = Path(sys.argv[protocol_index])
        if not output.is_file():
            raise
        gate = json.loads(output.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        one_step_auc = float(protocol["v7_one_step_active_auc"])
        two_step_auc = float(gate["active_cross_task_roc_auc"])
        gate["stage"] = "toolsandbox_v9_two_step_feasibility_gate_seed42"
        gate["feature_variant"] = "first_two_branch_receipts"
        gate["probe_horizon"] = 2
        gate["v7_one_step_active_auc"] = one_step_auc
        gate["two_step_auc_gain_over_v7"] = two_step_auc - one_step_auc
        gate["continuation_policy_calls_per_probed_event"] = 2
        gate["maximum_tool_executions_per_probed_event"] = 4
        gate["continuation_call_rate_per_event"] = 2 * float(gate["probe_rate"])
        gate["maximum_tool_execution_rate_per_event"] = 4 * float(
            gate["probe_rate"]
        )
        gate["outcome_checks"]["two_step_beats_one_step"] = (
            two_step_auc > one_step_auc + 1e-12
        )
        passed = all(gate["integrity_checks"].values()) and all(
            gate["outcome_checks"].values()
        )
        gate["passed"] = passed
        gate["feasibility_passed"] = passed
        gate["next_step"] = (
            "two-step offline signal passes; next validate live two-step probing"
            if passed
            else "two-step offline receipts do not clear the frozen feasibility gate"
        )
        output.write_text(
            json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
