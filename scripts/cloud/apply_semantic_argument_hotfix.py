#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def patch_file(relative: str, replacements: list[tuple[str, str]]) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements:
        if new in text:
            continue
        if old not in text:
            raise RuntimeError(f"cannot locate patch anchor in {relative}: {old[:80]!r}")
        text = text.replace(old, new, 1)
    if text == original:
        print(f"UNCHANGED {relative}")
        return
    backup = path.with_name(path.name + ".before_semantic_argument_hotfix")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"PATCHED {relative}")


patch_file(
    "environments/api_bank/harness.py",
    [
        (
            "from rescuecredit.types import HarnessDecision\n",
            "from rescuecredit.types import HarnessDecision\n\nfrom .adapter import canonical_action\n",
        ),
        (
            """                patch_id = "missing_required_argument"
                corrected = copy.deepcopy(proposal)
                expected_arguments = dict(expected.get("arguments", {}))
                corrected.setdefault("arguments", {})
                for key in required:
                    if key not in corrected["arguments"] and key in expected_arguments:
                        corrected["arguments"][key] = expected_arguments[key]
                deterministic = True
""",
            """                patch_id = "missing_required_argument"
                # A proposal may be missing one field and contain a wrong value
                # in another.  The controlled harness only applies corrections
                # that match the checker-approved action in full.
                corrected = copy.deepcopy(expected)
                deterministic = True
""",
        ),
        (
            """                deterministic = True

        if patch_id == "none" or self.condition == "H0":
""",
            """                deterministic = True
            elif canonical_action(proposal) != canonical_action(expected):
                # The tool/schema can be valid while one or more argument values
                # are semantically wrong.  Treat this as a teachable correction
                # instead of repeatedly executing a recoverable no-op.
                patch_id = "semantic_argument_mismatch"
                corrected = copy.deepcopy(expected)

        if patch_id == "none" or self.condition == "H0":
""",
        ),
        (
            '        event_type = "replace" if patch_id == "wrong_tool_replace" else "repair" if patch_id == "missing_required_argument" else "reject"\n',
            """        event_type = (
            "replace"
            if patch_id in {"wrong_tool_replace", "semantic_argument_mismatch"}
            else "repair"
            if patch_id == "missing_required_argument"
            else "reject"
        )
""",
        ),
    ],
)

patch_file(
    "rescuecredit/training.py",
    [
        (
            'TEACHABLE_PATCHES = {"missing_required_argument", "wrong_tool_replace", "premature_finish"}\n',
            """TEACHABLE_PATCHES = {
    "missing_required_argument",
    "wrong_tool_replace",
    "semantic_argument_mismatch",
    "premature_finish",
}
""",
        )
    ],
)

patch_file(
    "scripts/run_train.py",
    [
        (
            '                shadow_safe=decision.patch_id == "wrong_tool_replace",\n',
            '                shadow_safe=decision.patch_id in {"wrong_tool_replace", "semantic_argument_mismatch"},\n',
        )
    ],
)

patch_file(
    "scripts/run_eval.py",
    [
        (
            """        exact_first = expected is not None and canonical_action(proposal) == canonical_action(expected)
        first_pass_valid_calls += int(exact_first)
        executed, decision = harness.execute(env.observation(), proposal, expected)
        interventions += int(decision.triggered and decision.changes_execution)
""",
            """        proposal_matches_expected = expected is not None and canonical_action(proposal) == canonical_action(expected)
        first_pass_valid_calls += int(proposal_matches_expected)
        executed, decision = harness.execute(env.observation(), proposal, expected)
        executed_matches_expected = expected is not None and canonical_action(executed) == canonical_action(expected)
        interventions += int(decision.triggered and decision.changes_execution)
""",
        ),
        (
            '                "ground_truth_match": exact_first,\n',
            """                "ground_truth_match": executed_matches_expected,
                "proposal_ground_truth_match": proposal_matches_expected,
                "executed_ground_truth_match": executed_matches_expected,
                "patch_applied": decision.patch_id if decision.triggered and decision.changes_execution else None,
""",
        ),
    ],
)

patch_file(
    "environments/api_bank/adapter.py",
    [
        (
            """        reason = "running"
        tool_result: dict[str, Any] | None = None
        if action.get("type") == "finish":
""",
            """        reason = "running"
        tool_result: dict[str, Any] | None = None
        ground_truth_match = False
        if action.get("type") == "finish":
""",
        ),
        (
            """            reward = float(self.success)
            reason = "success" if self.success else "premature_finish"
""",
            """            reward = float(self.success)
            ground_truth_match = self.success
            reason = "success" if self.success else "premature_finish"
""",
        ),
        (
            """            if expected is not None and normalized == canonical_action(expected):
                self.calls.append(normalized)
""",
            """            if expected is not None and normalized == canonical_action(expected):
                ground_truth_match = True
                self.calls.append(normalized)
""",
        ),
        (
            '            "ground_truth_match": bool(reward),\n',
            '            "ground_truth_match": ground_truth_match,\n',
        ),
    ],
)

test_path = ROOT / "tests/test_semantic_argument_hotfix.py"
test_path.write_text(
    '''from environments.api_bank import APIBankControlledEnv, APIBankHarness
from rescuecredit.training import CreditAssigner


TASK = {
    "available_tools": [{"name": "SendEmail", "required": ["to", "body"], "optional": []}],
    "reference_actions": [{"tool": "SendEmail", "arguments": {"to": "a@b.com", "body": "hello"}}],
    "max_steps": 4,
}


def test_semantic_argument_mismatch_is_repaired_and_auditable():
    env = APIBankControlledEnv()
    env.reset(TASK, 9)
    bad = {"tool": "SendEmail", "arguments": {"to": "wrong", "body": "hello"}}
    executed, decision = APIBankHarness("H3").execute(env.observation(), bad, env.expected_action())
    assert decision.patch_id == "semantic_argument_mismatch"
    assert decision.triggered and decision.changes_execution
    assert executed == TASK["reference_actions"][0]
    _, _, _, info = env.step(executed)
    assert info["ground_truth_match"] is True

    record = CreditAssigner("rescuecredit", audit_probability=1.0).assign(bad, TASK["reference_actions"][0])
    assert record.intervened and record.teachable
    assert record.audit_draw == 1 and record.shadow_steps == 1


def test_missing_argument_repair_drops_other_wrong_values():
    env = APIBankControlledEnv()
    env.reset(TASK, 10)
    bad = {"tool": "SendEmail", "arguments": {"to": "wrong"}}
    executed, decision = APIBankHarness("H3").execute(env.observation(), bad, env.expected_action())
    assert decision.patch_id == "missing_required_argument"
    assert executed == TASK["reference_actions"][0]
    _, _, _, info = env.step(executed)
    assert info["ground_truth_match"] is True
''',
    encoding="utf-8",
)
print(f"WROTE {test_path.relative_to(ROOT)}")
print("SEMANTIC_ARGUMENT_HOTFIX_OK")
