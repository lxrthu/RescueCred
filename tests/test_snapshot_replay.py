import json

from environments.api_bank.adapter import APIBankControlledEnv
from environments.rescue_mdp.env import RescueMDP
from rescuecredit.logging import JsonlLogger
from rescuecredit.types import RescueEvent, TokenSpan


def test_toy_snapshot_replay_determinism():
    env = RescueMDP()
    env.reset(seed=7)
    reference = env.snapshot()
    rng = env.get_rng_state()
    first = env.step({"type": "correct_tool"})
    env.restore(reference)
    env.set_rng_state(rng)
    second = env.step({"type": "correct_tool"})
    assert first == second


def test_api_bank_snapshot_replay_determinism():
    task = {
        "task_id": "t",
        "user_goal": "g",
        "available_tools": [{"name": "Tool", "required": ["x"], "optional": []}],
        "reference_actions": [{"tool": "Tool", "arguments": {"x": 1}}],
    }
    env = APIBankControlledEnv()
    env.reset(task, 11)
    reference = env.snapshot()
    rng = env.get_rng_state()
    first = env.step({"tool": "Tool", "arguments": {"x": 1}})
    env.restore(reference)
    env.set_rng_state(rng)
    second = env.step({"tool": "Tool", "arguments": {"x": 1}})
    assert first == second


def test_api_bank_receipt_is_private_until_matching_tool_executes():
    task = {
        "task_id": "token-task",
        "user_goal": "Authenticate user3.",
        "available_tools": [{"name": "GetUserToken", "required": ["username", "password"], "optional": []}],
        "reference_actions": [
            {"tool": "GetUserToken", "arguments": {"username": "user3", "password": "user3pass"}}
        ],
        "reference_tool_receipts": [
            {"status": "ok", "tool": "GetUserToken", "token": "runtime-token"}
        ],
    }
    env = APIBankControlledEnv()
    observation = env.reset(task, 11)
    assert "reference_tool_receipts" not in observation
    assert "runtime-token" not in json.dumps(observation)
    _, _, _, info = env.step(
        {"tool": "GetUserToken", "arguments": {"username": "user3", "password": "user3pass"}}
    )
    assert info["tool_result"]["token"] == "runtime-token"


def test_api_bank_snapshot_restores_in_new_process_equivalent(tmp_path):
    task = {
        "task_id": "t",
        "user_goal": "g",
        "available_tools": [{"name": "Tool", "required": ["x"], "optional": []}],
        "reference_actions": [{"tool": "Tool", "arguments": {"x": 1}}],
    }
    first_env = APIBankControlledEnv(snapshot_dir=tmp_path)
    first_env.reset(task, 11)
    reference = first_env.snapshot()
    first = first_env.step({"tool": "Tool", "arguments": {"x": 1}})
    second_env = APIBankControlledEnv(snapshot_dir=tmp_path)
    second_env.restore(reference)
    second = second_env.step({"tool": "Tool", "arguments": {"x": 1}})
    assert first == second


def test_rescue_event_jsonl_restores_persistent_snapshot(tmp_path):
    task = {
        "task_id": "event-task",
        "source_sample_id": "source-1",
        "user_goal": "g",
        "available_tools": [{"name": "Tool", "required": ["x"], "optional": []}],
        "reference_actions": [{"tool": "Tool", "arguments": {"x": 1}}],
    }
    snapshot_dir = tmp_path / "snapshots"
    env = APIBankControlledEnv(snapshot_dir=snapshot_dir)
    observation = env.reset(task, 17)
    state_ref = env.snapshot()
    event = RescueEvent(
        run_id="run",
        episode_id="episode",
        group_id="group",
        candidate_id="candidate",
        step_id=0,
        state_ref=state_ref,
        state_hash=observation["state_hash"],
        proposal_text="{}",
        proposal_action={"tool": "Tool", "arguments": {}},
        executed_action={"tool": "Tool", "arguments": {"x": 1}},
        correction_text=None,
        event_type="repair",
        patch_id="missing_required_argument",
        patch_version="controlled-v1",
        verifier_label=0.0,
        verifier_confidence=1.0,
        verifier_reason="test",
        deterministic_outcome=False,
        shadow_safe=False,
        teachable_patch=True,
        permanent_safety_patch=False,
        intervention_step=0,
        token_spans=[TokenSpan(0, 1, "policy", "prefix")],
        metadata={"snapshot_digest": state_ref.split(":", 1)[1], "generation_seed": 17},
    )
    log_path = tmp_path / "events.jsonl"
    JsonlLogger(log_path).write(event)
    persisted = json.loads(log_path.read_text(encoding="utf-8"))

    replay_env = APIBankControlledEnv(snapshot_dir=snapshot_dir)
    replay_env.restore(persisted["state_ref"])
    assert replay_env.observation()["state_hash"] == persisted["state_hash"]
    assert persisted["metadata"]["snapshot_digest"] == persisted["state_ref"].split(":", 1)[1]
