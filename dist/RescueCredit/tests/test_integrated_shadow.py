from environments.api_bank import APIBankControlledEnv, APIBankHarness
from environments.api_bank.shadow import APIBankShadowRunner


TASK = {
    "task_id": "t",
    "user_goal": "send a message",
    "available_tools": [{"name": "SendEmail", "required": ["to", "body"], "optional": []}],
    "reference_actions": [{"tool": "SendEmail", "arguments": {"to": "a@b.com", "body": "hello"}}],
    "max_steps": 4,
}


def test_h3_main_succeeds_but_original_action_shadow_fails():
    env = APIBankControlledEnv()
    env.reset(TASK, 42)
    obs = env.observation()
    state_ref = env.snapshot()
    rng_state = env.get_rng_state()
    original = {"tool": "SendEmail", "arguments": {"to": "a@b.com"}}
    executed, decision = APIBankHarness("H3").execute(obs, original, env.expected_action())
    assert decision.patch_id == "missing_required_argument"
    _, reward, done, _ = env.step(executed)
    assert not done and reward == 0.0
    _, reward, done, _ = env.step({"type": "finish"})
    assert done and reward == 1.0

    shadow = APIBankShadowRunner(env).run(
        state_ref,
        original,
        decision.patch_id,
        rng_state,
        max_steps=4,
        expected_state_hash=obs["state_hash"],
        continuation=lambda *_: {"type": "finish"},
    )
    assert shadow.replay_valid
    assert shadow.task_success is False
    assert shadow.return_value == 0.0


def test_schema_valid_off_path_call_can_recover_later():
    task = {
        "task_id": "recover",
        "user_goal": "send",
        "available_tools": [
            {"name": "Lookup", "required": ["query"], "optional": []},
            {"name": "SendEmail", "required": ["to", "body"], "optional": []},
        ],
        "reference_actions": [{"tool": "SendEmail", "arguments": {"to": "a@b.com", "body": "hello"}}],
        "max_steps": 4,
    }
    env = APIBankControlledEnv()
    env.reset(task, 3)
    _, reward, done, info = env.step({"tool": "Lookup", "arguments": {"query": "a"}})
    assert not done and reward == 0.0
    assert info["tool_result"]["status"] == "no_effect"
    env.step(task["reference_actions"][0])
    _, reward, done, _ = env.step({"type": "finish"})
    assert done and reward == 1.0


def test_shadow_horizon_is_censored_not_valid_zero():
    env = APIBankControlledEnv(max_steps=5)
    env.reset(TASK, 5)
    reference = env.snapshot()
    shadow = APIBankShadowRunner(env).run(
        reference,
        {"tool": "SendEmail", "arguments": {"to": "wrong", "body": "still-valid"}},
        "wrong_tool_replace",
        env.get_rng_state(),
        max_steps=1,
        expected_state_hash=env.observation()["state_hash"],
        continuation=lambda *_: {"type": "finish"},
    )
    assert shadow.terminal_reason == "shadow_horizon"
    assert shadow.replay_valid is False
