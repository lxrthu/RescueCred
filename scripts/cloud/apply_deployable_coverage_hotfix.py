#!/usr/bin/env python3
"""Apply the deployable-harness natural-error coverage hotfix in place.

This is intentionally dependency-free so it can be pasted onto a passwordless
GPU server. Every edit is guarded and idempotent.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"ALREADY_PATCHED {relative}")
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one patch anchor in {relative}, found {count}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"PATCHED {relative}")


training_path = ROOT / "rescuecredit/training.py"
training_text = training_path.read_text(encoding="utf-8")
if "def history_patch_id(" not in training_text:
    old_training_anchor = "    return None\n\n\ndef group_normalize"
    new_training_anchor = '''    return None


def history_patch_id(decision: Any) -> str | None:
    """Expose only execution-changing patches to subsequent policy prompts."""

    if decision.triggered and decision.changes_execution:
        return str(decision.patch_id)
    return None


def group_normalize'''
    if training_text.count(old_training_anchor) != 1:
        raise RuntimeError("training history helper patch anchor missing")
    training_path.write_text(
        training_text.replace(old_training_anchor, new_training_anchor, 1),
        encoding="utf-8",
    )
    print("PATCHED rescuecredit/training.py")
else:
    print("ALREADY_PATCHED rescuecredit/training.py")

replace_once(
    "environments/api_bank/correction_generator.py",
    '''        "3. Only fill missing required arguments.\\n"
        "4. Never invent a value: copy values from the user goal or prior tool receipt.\\n"
        "5. If the missing value is ambiguous or unavailable, output {}.\\n\\n"''',
    '''        "3. Fill missing required arguments and replace an existing argument only when its current value is unsupported or contradicts visible context.\\n"
        "4. Preserve every already-supported argument exactly.\\n"
        "5. Never invent a value: copy exact values from the user goal or prior tool receipt.\\n"
        "6. If any repair value is ambiguous or unavailable, output {}.\\n\\n"''',
)

replace_once(
    "environments/api_bank/deployable.py",
    '''            and not a_validity.executable_valid
            and "missing required arguments" in a_validity.reason''',
    '''            and schema is not None
            and a_validity.semantic_valid != "true"''',
)

replace_once(
    "environments/api_bank/deployable.py",
    '''            "generated_visible_schema_repair"
            if generated_repair
            else "visible_schema_repair"''',
    '''            "generated_visible_schema_repair"
            if generated_repair and missing_before
            else "generated_visible_argument_repair"
            if generated_repair
            else "visible_schema_repair"''',
)

replace_once(
    "scripts/run_eval.py",
    "from rescuecredit.training import build_prompt, parse_action",
    "from rescuecredit.training import build_prompt, history_patch_id, parse_action",
)
replace_once(
    "scripts/run_eval.py",
    "    interventions = 0\n    reward = 0.0",
    "    interventions = 0\n    feedback_events = 0\n    reward = 0.0",
)
replace_once(
    "scripts/run_eval.py",
    "        interventions += int(decision.triggered and decision.changes_execution)",
    "        feedback_events += int(decision.triggered)\n        interventions += int(decision.triggered and decision.changes_execution)",
)
replace_once(
    "scripts/run_eval.py",
    '                "harness_patch": decision.patch_id if decision.triggered else None,',
    '                "harness_patch": history_patch_id(decision),',
)
replace_once(
    "scripts/run_eval.py",
    '''        "intervened": interventions > 0,
        "interventions": interventions,
        "environment_steps": env.steps,''',
    '''        "intervened": interventions > 0,
        "interventions": interventions,
        "feedback_triggered": feedback_events > 0,
        "feedback_events": feedback_events,
        "environment_steps": env.steps,''',
)
replace_once(
    "scripts/run_eval.py",
    '''            "intervened": on["intervened"],
            "harness_on_steps": on["environment_steps"],''',
    '''            "intervened": on["intervened"],
            "feedback_triggered": on["feedback_triggered"],
            "feedback_events": on["feedback_events"],
            "execution_interventions": on["interventions"],
            "harness_on_steps": on["environment_steps"],''',
)
replace_once(
    "scripts/run_eval.py",
    '''        "intervention_rate": sum(record["intervened"] for record in records) / count,
        "evaluation_steps":''',
    '''        "intervention_rate": sum(record["intervened"] for record in records) / count,
        "feedback_task_rate": sum(record["feedback_triggered"] for record in records) / count,
        "feedback_events": sum(record["feedback_events"] for record in records),
        "execution_interventions": sum(record["execution_interventions"] for record in records),
        "evaluation_steps":''',
)

replace_once(
    "scripts/run_train.py",
    "from rescuecredit.training import build_prompt, group_normalize, parse_action",
    "from rescuecredit.training import build_prompt, group_normalize, history_patch_id, parse_action",
)
replace_once(
    "scripts/run_train.py",
    '''                    "generated_visible_schema_repair",
                },''',
    '''                    "generated_visible_schema_repair",
                    "generated_visible_argument_repair",
                },''',
)

train_path = ROOT / "scripts/run_train.py"
train_text = train_path.read_text(encoding="utf-8")
old_history = '"harness_patch": decision.patch_id if decision.triggered else None'
new_history = '"harness_patch": history_patch_id(decision)'
if old_history in train_text:
    replaced = train_text.count(old_history)
    train_path.write_text(train_text.replace(old_history, new_history), encoding="utf-8")
    print(f"PATCHED scripts/run_train.py history_sites={replaced}")
elif new_history in train_text:
    print("ALREADY_PATCHED scripts/run_train.py history_sites")
else:
    raise RuntimeError("run_train history patch anchor missing")


def self_check() -> None:
    from environments.api_bank import DeployableAPIBankHarness
    from rescuecredit.training import history_patch_id

    observation = {
        "user_goal": "Cancel appointment 56789012.",
        "available_tools": [
            {"name": "CancelRegistration", "required": ["appointment_id"], "optional": []}
        ],
        "success_predicate_satisfied": False,
    }

    def generator(_observation, _proposal, _reason, _receipt):
        return {"tool": "CancelRegistration", "arguments": {"appointment_id": "56789012"}}

    proposal = {"tool": "CancelRegistration", "arguments": {"appointment_id": "bad-id"}}
    executed, decision = DeployableAPIBankHarness("H3", correction_generator=generator).execute(
        observation, proposal
    )
    assert decision.patch_id == "generated_visible_argument_repair"
    assert decision.changes_execution
    assert executed["arguments"]["appointment_id"] == "56789012"

    unresolved_observation = {
        "user_goal": "Please authenticate me.",
        "available_tools": [
            {"name": "GetUserToken", "required": ["username", "password"], "optional": []}
        ],
        "success_predicate_satisfied": False,
    }
    _, unresolved = DeployableAPIBankHarness("H3").execute(
        unresolved_observation,
        {"tool": "GetUserToken", "arguments": {"username": "admin"}},
    )
    assert unresolved.triggered and not unresolved.changes_execution
    assert history_patch_id(unresolved) is None


self_check()
print("DEPLOYABLE_COVERAGE_HOTFIX_OK")
