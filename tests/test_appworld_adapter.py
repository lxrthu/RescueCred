import copy
import json

import pytest

from environments.appworld.adapter import (
    AppWorldAtomicEnv,
    canonical_appworld_action,
    normalize_function_tools,
    render_atomic_call,
)


class FakeDocs:
    def function_calling(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "mail__send_email",
                    "description": "Send an email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "body"],
                    },
                },
            }
        ]


class FakeTask:
    instruction = "Send hello to a@example.com"
    api_docs = FakeDocs()


class FakeWorld:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.task = FakeTask()
        self.counter = 0
        self.states = {}
        self.environment_io = []
        self.num_interactions = 0
        self.num_sub_interactions = 0

    def execute(self, code):
        self.counter += 1
        self.num_interactions += 1
        self.num_sub_interactions += 1
        self.environment_io.append(code)
        return json.dumps({"ok": True, "counter": self.counter, "code": code})

    def save_state(self):
        state_id = f"state-{len(self.states)}"
        self.states[state_id] = self.counter
        return state_id

    def load_state(self, state_id):
        self.counter = self.states[state_id]

    def export_state(self):
        return {"counter": self.counter}

    def task_completed(self):
        return False

    def save(self):
        return None

    def evaluate(self):
        return {"success": self.counter == 1}

    def close(self):
        return None


def test_function_docs_normalize_without_ground_truth():
    tools = normalize_function_tools(FakeDocs())
    assert tools[0]["name"] == "mail__send_email"
    assert tools[0]["required"] == ["to", "body"]
    assert tools[0]["optional"] == []


def test_atomic_call_rejects_identifier_injection_and_quotes_arguments():
    action = {
        "tool": "mail__send_email",
        "arguments": {"to": "a@example.com", "body": "'); import os; #"},
    }
    code = render_atomic_call(action)
    assert "apis.mail.send_email" in code
    assert "json.loads" in code
    with pytest.raises(ValueError):
        render_atomic_call({"app": "mail;import os", "api": "send", "arguments": {}})


def test_snapshot_restores_appworld_state_transcript_and_rng():
    env = AppWorldAtomicEnv(world_factory=FakeWorld, max_steps=4)
    observation = env.reset({"task_id": "train-1"}, seed=7)
    assert observation["reference_free_observation"] is True
    assert "ground_truth" not in observation
    state_ref = env.snapshot()
    rng_state = copy.deepcopy(env.get_rng_state())
    action = {
        "tool": "mail__send_email",
        "arguments": {"to": "a@example.com", "body": "hi"},
    }
    first = env.step(action)
    assert env.world.num_interactions == 1
    env.restore(state_ref)
    assert env.world.num_interactions == 0
    assert env.world.num_sub_interactions == 0
    assert env.world.environment_io == []
    env.set_rng_state(rng_state)
    second = env.step(action)
    assert first == second
    assert env.task_success() is True


def test_canonical_action_accepts_explicit_app_api():
    assert canonical_appworld_action(
        {"app": "spotify", "api": "show_song", "arguments": {"song_id": 1}}
    ) == {"tool": "spotify__show_song", "arguments": {"song_id": 1}}
