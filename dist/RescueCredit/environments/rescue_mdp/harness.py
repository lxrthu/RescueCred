from __future__ import annotations

from rescuecredit.types import HarnessDecision


class RescueMDPHarness:
    CONDITIONS = {"H0", "H1", "H2", "H3", "Hplacebo"}

    def __init__(self, condition: str) -> None:
        if condition not in self.CONDITIONS:
            raise ValueError(f"unknown harness condition {condition}")
        self.condition = condition

    def inspect(self, obs: dict, proposal: dict) -> HarnessDecision:
        stage, name = obs["stage"], proposal.get("type")
        correction = None
        patch_id = "none"
        event_type = "feedback"
        deterministic = True
        if stage == "choose_tool" and name == "wrong_tool":
            patch_id, correction, event_type, deterministic = "wrong_tool_replace", {"type": "correct_tool"}, "replace", False
        elif stage == "fill_args" and name == "missing_argument":
            patch_id, correction, event_type = "missing_required_argument", {"type": "valid_call"}, "repair"
        elif name == "premature_finish":
            patch_id, event_type = "premature_finish", "reject"
            correction = {"type": "correct_tool" if stage == "choose_tool" else "valid_call" if stage == "fill_args" else "finish"}
        else:
            return HarnessDecision(False, "feedback", "none", None, None, True, False, False)

        if self.condition == "H0":
            return HarnessDecision(False, event_type, patch_id, None, None, True, False, False, deterministic)
        if self.condition == "H1":
            return HarnessDecision(True, "feedback", patch_id, None, f"invalid action: {patch_id}", True, False, False, deterministic)
        if self.condition == "Hplacebo":
            return HarnessDecision(True, "feedback", "placebo", None, "Please verify the action.", True, False, False, deterministic)
        if self.condition == "H2":
            return HarnessDecision(True, "reject", patch_id, correction, "Rejected; retry using verifier feedback.", True, False, True, deterministic)
        return HarnessDecision(True, event_type, patch_id, correction, "Harness applied deterministic correction.", True, False, True, deterministic)

    def execute(self, obs: dict, proposal: dict) -> tuple[dict, HarnessDecision]:
        decision = self.inspect(obs, proposal)
        if self.condition == "H3" and decision.corrected_action is not None:
            return decision.corrected_action, decision
        return proposal, decision
