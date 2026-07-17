from __future__ import annotations

from rescuecredit.types import VerificationResult


class APIBankVerifier:
    """Verifier with deliberately narrow deterministic scope."""

    def verify(self, state: dict, action: dict) -> VerificationResult:
        if action.get("type") == "finish":
            valid = bool(state.get("success_predicate_satisfied"))
            return VerificationResult(valid, float(valid), 1.0, True, "task state predicate")
        tool = str(action.get("tool", ""))
        arguments = dict(action.get("arguments", {}))
        catalog = {entry["name"]: entry for entry in state.get("available_tools", [])}
        if tool not in catalog:
            return VerificationResult(False, 0.0, 1.0, True, "tool not in action set")
        required = set(catalog[tool].get("required", []))
        missing = sorted(required - arguments.keys())
        if missing:
            return VerificationResult(False, 0.0, 1.0, True, f"missing required arguments: {missing}")
        # Schema validity cannot prove long-horizon semantic success.
        return VerificationResult(True, 0.5, 0.6, False, "schema valid; semantic outcome requires shadow/checker")

