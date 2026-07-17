#!/usr/bin/env python3
"""Keep the complete audited counterfactual continuation Harness-off (H0)."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"ALREADY_PATCHED {relative}")
        return
    if text.count(old) != 1:
        raise RuntimeError(f"H0 shadow anchor mismatch in {relative}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"PATCHED {relative}")


replace_once("scripts/run_train.py", "import argparse\nimport copy\nimport contextlib", "import argparse\nimport contextlib")
replace_once(
    "scripts/run_train.py",
    "    history_before_event: list[dict[str, Any]] = []\n    tool_context_before_event: dict[str, Any] | None = None\n    reward = 0.0",
    "    history_before_event: list[dict[str, Any]] = []\n    reward = 0.0",
)
replace_once(
    "scripts/run_train.py",
    "            history_before_event = list(history)\n            tool_context_before_event = copy.deepcopy(previous_tool_result)",
    "            history_before_event = list(history)",
)
replace_once(
    "scripts/run_train.py",
    "        shadow_history = list(history_before_event)\n        shadow_tool_context = copy.deepcopy(tool_context_before_event)",
    "        shadow_history = list(history_before_event)",
)
replace_once(
    "scripts/run_train.py",
    '''        shadow_oracle_harness = OracleAPIBankHarness("H3")

        def continuation(obs: dict, shadow_step: int, _disabled_patch_id: str, previous_tool_result: dict | None) -> dict:
            nonlocal pending_history, shadow_tool_context
            shadow_tool_context = merge_visible_tool_context(
                shadow_tool_context, previous_tool_result
            )''',
    '''        def continuation(obs: dict, shadow_step: int, _disabled_patch_id: str, previous_tool_result: dict | None) -> dict:
            nonlocal pending_history''',
)
replace_once(
    "scripts/run_train.py",
    '''            executed, decision, _, _ = execute_harness_action(
                args.method,
                env,
                obs,
                proposal,
                shadow_tool_context,
                shadow_oracle_harness,
                deployable_harness,
            )
            pending_history = {
                "policy_proposal": proposal,
                "harness_patch": history_patch_id(decision),
                "executed_action": executed,
                "tool_result": None,
            }
            return executed''',
    '''            pending_history = {
                "policy_proposal": proposal,
                "harness_patch": None,
                "executed_action": proposal,
                "tool_result": None,
            }
            # G0 is the fully unassisted potential return from the committed
            # pre-intervention state. Re-enabling the Harness here would map
            # both potential outcomes back to success and erase causal credit.
            return proposal''',
)

replace_once(
    "scripts/evaluate_full_shadow.py",
    '''    shadow_harness = APIBankHarness("H3")

    def continuation(obs: dict, step: int, _disabled: str, previous_tool_result: dict | None) -> dict:''',
    '''    def continuation(obs: dict, step: int, _disabled: str, previous_tool_result: dict | None) -> dict:''',
)
replace_once(
    "scripts/evaluate_full_shadow.py",
    '''        expected = env.expected_action()
        executed, decision = shadow_harness.execute(obs, proposal, expected)
        pending = {"policy_proposal": proposal, "executed_action": executed, "tool_result": None}
        return executed''',
    '''        pending = {"policy_proposal": proposal, "executed_action": proposal, "tool_result": None}
        return proposal''',
)


train = (ROOT / "scripts/run_train.py").read_text(encoding="utf-8")
shadow = train.split("    def shadow_factory():", 1)[1].split("    if args.method ==", 1)[0]
assert "execute_harness_action(" not in shadow
assert '"harness_patch": None' in shadow
assert "return proposal" in shadow

full = (ROOT / "scripts/evaluate_full_shadow.py").read_text(encoding="utf-8")
continuation = full.split("    def continuation(", 1)[1].split("    shadow =", 1)[0]
assert "shadow_harness.execute" not in continuation
assert "return proposal" in continuation

print("H0_SHADOW_HOTFIX_OK")
