from __future__ import annotations

import copy

from rescuecredit.types import HarnessDecision

from .adapter import canonical_action


class OracleAPIBankHarness:
    """Reference-backed diagnostic harness.

    This harness intentionally receives the frozen expected action.  It is an
    oracle upper bound and must not be used as the paper's deployable harness.
    """
    CONDITIONS = {"H0", "H1", "H2", "H3", "Hplacebo"}

    def __init__(self, condition: str) -> None:
        if condition not in self.CONDITIONS:
            raise ValueError(f"unknown condition {condition}")
        self.condition = condition

    def inspect(self, obs: dict, proposal: dict, expected: dict | None) -> HarnessDecision:
        patch_id = "none"
        corrected = None
        deterministic = False
        if proposal.get("type") == "finish" and not obs.get("success_predicate_satisfied"):
            patch_id = "premature_finish"
            corrected = copy.deepcopy(expected)
            deterministic = True
        elif expected is not None and isinstance(proposal.get("tool"), str) and proposal.get("tool") and proposal.get("tool") != expected.get("tool"):
            patch_id = "wrong_tool_replace"
            corrected = copy.deepcopy(expected)
        elif expected is not None:
            required = set()
            for tool in obs.get("available_tools", []):
                if tool.get("name") == proposal.get("tool"):
                    required = set(tool.get("required", []))
                    break
            if required - set(dict(proposal.get("arguments", {}))):
                patch_id = "missing_required_argument"
                # A proposal may be missing one field and contain a wrong value
                # in another.  The controlled harness only applies corrections
                # that match the checker-approved action in full.
                corrected = copy.deepcopy(expected)
                deterministic = True
            elif canonical_action(proposal) != canonical_action(expected):
                # The tool/schema can be valid while one or more argument values
                # are semantically wrong.  Treat this as a teachable correction
                # instead of repeatedly executing a recoverable no-op.
                patch_id = "semantic_argument_mismatch"
                corrected = copy.deepcopy(expected)

        if patch_id == "none" or self.condition == "H0":
            return HarnessDecision(False, "feedback", patch_id, None, None, True, False, False, deterministic)
        if self.condition == "Hplacebo":
            return HarnessDecision(True, "feedback", "placebo", None, "Please inspect this action.", True, False, False, deterministic)
        if self.condition == "H1":
            return HarnessDecision(True, "feedback", patch_id, None, f"Verifier flagged {patch_id}", True, False, False, deterministic)
        if self.condition == "H2":
            return HarnessDecision(True, "reject", patch_id, corrected, f"Rejected because {patch_id}", True, False, True, deterministic)
        event_type = (
            "replace"
            if patch_id in {"wrong_tool_replace", "semantic_argument_mismatch"}
            else "repair"
            if patch_id == "missing_required_argument"
            else "reject"
        )
        return HarnessDecision(True, event_type, patch_id, corrected, f"Applied {patch_id}", True, False, True, deterministic)

    def execute(self, obs: dict, proposal: dict, expected: dict | None) -> tuple[dict, HarnessDecision]:
        decision = self.inspect(obs, proposal, expected)
        if self.condition == "H3" and decision.corrected_action is not None:
            return decision.corrected_action, decision
        return proposal, decision


# Backward-compatible name for the already reported Oracle-Teacher runs.
# New paper-facing code must import DeployableAPIBankHarness explicitly.
APIBankHarness = OracleAPIBankHarness
