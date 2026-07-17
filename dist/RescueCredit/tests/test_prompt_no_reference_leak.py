from rescuecredit.training import build_prompt


def test_prompt_does_not_teacher_force_reference_actions():
    task = {
        "user_goal": "public goal",
        "available_tools": [{"name": "Tool", "required": []}],
        "reference_actions": [{"tool": "SECRET_REFERENCE_TOOL", "arguments": {"secret": "DO_NOT_LEAK"}}],
    }
    prompt = build_prompt(task, [])
    assert "SECRET_REFERENCE_TOOL" not in prompt
    assert "DO_NOT_LEAK" not in prompt

