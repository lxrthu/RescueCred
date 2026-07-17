from __future__ import annotations

import copy
import hashlib
import json
import random
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class MDPState:
    stage: str = "choose_tool"
    correct_tool: bool = False
    valid_arguments: bool = False
    success: bool = False
    done: bool = False
    steps: int = 0


class RescueMDP:
    """Small deterministic schema-rescue MDP with snapshot and RNG replay."""

    ACTIONS = {
        "choose_tool": ("correct_tool", "wrong_tool", "premature_finish"),
        "fill_args": ("valid_call", "missing_argument", "wrong_tool", "premature_finish"),
        "tool_response": ("finish", "premature_finish"),
    }

    def __init__(self, max_steps: int = 6) -> None:
        self.max_steps = max_steps
        self.rng = random.Random()
        self.state = MDPState()
        self._snapshots: dict[str, tuple[MDPState, object]] = {}

    def reset(self, task: dict[str, Any] | None = None, seed: int = 0) -> dict[str, Any]:
        del task
        self.rng.seed(seed)
        self.state = MDPState()
        self._snapshots.clear()
        return self.observation()

    def observation(self) -> dict[str, Any]:
        return {**self.state.__dict__, "state_hash": self.state_hash()}

    def legal_actions(self) -> tuple[str, ...]:
        return self.ACTIONS.get(self.state.stage, ())

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.state.done:
            raise RuntimeError("episode already terminated")
        name = str(action.get("type", ""))
        self.state.steps += 1
        reward = 0.0
        terminal_reason = "running"

        if self.state.stage == "choose_tool":
            if name == "correct_tool":
                self.state.correct_tool = True
                self.state.stage = "fill_args"
            else:
                self.state.done = True
                terminal_reason = "invalid_initial_action"
        elif self.state.stage == "fill_args":
            if name == "valid_call" and self.state.correct_tool:
                self.state.valid_arguments = True
                self.state.stage = "tool_response"
            else:
                self.state.done = True
                terminal_reason = "invalid_tool_call"
        elif self.state.stage == "tool_response":
            self.state.done = True
            if name == "finish" and self.state.correct_tool and self.state.valid_arguments:
                self.state.success = True
                reward = 1.0
                terminal_reason = "success"
            else:
                terminal_reason = "premature_finish"
        else:
            raise RuntimeError(f"unknown stage: {self.state.stage}")

        if self.state.steps >= self.max_steps and not self.state.done:
            self.state.done = True
            terminal_reason = "max_steps"
        return self.observation(), reward, self.state.done, {"terminal_reason": terminal_reason}

    def snapshot(self) -> str:
        reference = f"toy:{uuid.uuid4().hex}"
        self._snapshots[reference] = (copy.deepcopy(self.state), self.rng.getstate())
        return reference

    def restore(self, state_ref: str) -> None:
        state, rng_state = self._snapshots[state_ref]
        self.state = copy.deepcopy(state)
        self.rng.setstate(rng_state)

    def get_rng_state(self) -> object:
        return self.rng.getstate()

    def set_rng_state(self, rng_state: object) -> None:
        self.rng.setstate(rng_state)

    def task_success(self) -> bool:
        return self.state.success

    def state_hash(self) -> str:
        payload = json.dumps(self.state.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def clone(self) -> "RescueMDP":
        other = RescueMDP(self.max_steps)
        other.state = copy.deepcopy(self.state)
        other.rng.setstate(self.rng.getstate())
        other._snapshots = copy.deepcopy(self._snapshots)
        return other

