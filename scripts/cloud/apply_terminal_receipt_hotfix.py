#!/usr/bin/env python3
"""Install persistent visible receipts and state-based task termination."""

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
        raise RuntimeError(f"patch anchor mismatch in {relative}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"PATCHED {relative}")


deployable = ROOT / "environments/api_bank/deployable.py"
text = deployable.read_text(encoding="utf-8")
if "def merge_visible_tool_context(" not in text:
    marker = "\n\nclass VisibleContextSemanticValidator:"
    helper = '''

def merge_visible_tool_context(
    context: dict[str, Any] | None,
    latest_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Retain successful visible receipts across later failed/no-op calls."""

    merged = copy.deepcopy(context or {})
    if not latest_result:
        return merged or None
    latest = copy.deepcopy(latest_result)
    if latest.get("status") == "ok":
        merged.update(latest)
        merged.pop("latest_feedback", None)
        return merged
    if not merged:
        return latest
    merged["latest_feedback"] = latest
    if "suggested_action" in latest:
        merged["suggested_action"] = copy.deepcopy(latest["suggested_action"])
    return merged
'''
    if text.count(marker) != 1:
        raise RuntimeError("deployable helper anchor mismatch")
    deployable.write_text(text.replace(marker, helper + marker, 1), encoding="utf-8")
    print("PATCHED environments/api_bank/deployable.py")
else:
    print("ALREADY_PATCHED environments/api_bank/deployable.py")

replace_once(
    "environments/api_bank/__init__.py",
    "    VisibleContextSemanticValidator,\n    public_harness_observation,",
    "    VisibleContextSemanticValidator,\n    merge_visible_tool_context,\n    public_harness_observation,",
)
replace_once(
    "environments/api_bank/__init__.py",
    '    "ActionValidity",\n    "public_harness_observation",',
    '    "ActionValidity",\n    "merge_visible_tool_context",\n    "public_harness_observation",',
)

replace_once(
    "environments/api_bank/adapter.py",
    '''                else:
                    tool_result = {"status": "ok", "call_index": self.call_index, "tool": normalized["tool"]}
            elif self._is_executable(normalized):''',
    '''                else:
                    tool_result = {"status": "ok", "call_index": self.call_index, "tool": normalized["tool"]}
                if self.call_index == len(self.task.get("reference_actions", [])):
                    self.done = True
                    self.success = True
                    reward = 1.0
                    reason = "success"
            elif self._is_executable(normalized):''',
)

replace_once(
    "scripts/run_eval.py",
    "    OracleAPIBankHarness,\n    public_harness_observation,",
    "    OracleAPIBankHarness,\n    merge_visible_tool_context,\n    public_harness_observation,",
)
replace_once(
    "scripts/run_eval.py",
    '        previous_tool_result = info.get("tool_result")',
    '''        previous_tool_result = merge_visible_tool_context(
            previous_tool_result, info.get("tool_result")
        )''',
)

replace_once("scripts/run_train.py", "import argparse\nimport contextlib", "import argparse\nimport copy\nimport contextlib")
replace_once(
    "scripts/run_train.py",
    "    OracleAPIBankHarness,\n    public_harness_observation,",
    "    OracleAPIBankHarness,\n    merge_visible_tool_context,\n    public_harness_observation,",
)
replace_once(
    "scripts/run_train.py",
    "    history_before_event: list[dict[str, Any]] = []\n    reward = 0.0",
    "    history_before_event: list[dict[str, Any]] = []\n    tool_context_before_event: dict[str, Any] | None = None\n    reward = 0.0",
)
replace_once(
    "scripts/run_train.py",
    "            event_rng_state = rng_state\n            history_before_event = list(history)",
    "            event_rng_state = rng_state\n            history_before_event = list(history)\n            tool_context_before_event = copy.deepcopy(previous_tool_result)",
)
replace_once(
    "scripts/run_train.py",
    '        previous_tool_result = info.get("tool_result")',
    '''        previous_tool_result = merge_visible_tool_context(
            previous_tool_result, info.get("tool_result")
        )''',
)
replace_once(
    "scripts/run_train.py",
    "        shadow_history = list(history_before_event)\n        pending_history:",
    "        shadow_history = list(history_before_event)\n        shadow_tool_context = copy.deepcopy(tool_context_before_event)\n        pending_history:",
)
replace_once(
    "scripts/run_train.py",
    '''            nonlocal pending_history
            pending_history["tool_result"] = previous_tool_result''',
    '''            nonlocal pending_history, shadow_tool_context
            shadow_tool_context = merge_visible_tool_context(
                shadow_tool_context, previous_tool_result
            )
            pending_history["tool_result"] = previous_tool_result''',
)
replace_once(
    "scripts/run_train.py",
    '''                proposal,
                previous_tool_result,
                shadow_oracle_harness,''',
    '''                proposal,
                shadow_tool_context,
                shadow_oracle_harness,''',
)


def self_check() -> None:
    from environments.api_bank import APIBankControlledEnv, merge_visible_tool_context

    context = merge_visible_tool_context(None, {"status": "ok", "token": "abc123"})
    context = merge_visible_tool_context(context, {"status": "no_effect", "reason": "off path"})
    assert context and context["token"] == "abc123"

    task = {
        "task_id": "terminal-check",
        "user_goal": "cancel 56789012",
        "available_tools": [{"name": "Cancel", "required": ["appointment_id"], "optional": []}],
        "reference_actions": [{"tool": "Cancel", "arguments": {"appointment_id": "56789012"}}],
        "max_steps": 4,
    }
    env = APIBankControlledEnv()
    env.reset(task, 42)
    _, reward, done, info = env.step(task["reference_actions"][0])
    assert done and reward == 1.0 and info["terminal_reason"] == "success"


self_check()
print("TERMINAL_RECEIPT_HOTFIX_OK")
