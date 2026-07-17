from __future__ import annotations

import copy
import hashlib
import json
import random
import uuid
from pathlib import Path
from typing import Any


def canonical_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool": str(action.get("tool", "")),
        "arguments": {str(key): value for key, value in sorted(dict(action.get("arguments", {})).items())},
    }


class APIBankControlledEnv:
    """Serializable API-Bank-derived task state with a programmatic action checker.

    This deliberately reports controlled-environment scores, not official API-Bank
    leaderboard scores. A successful episode must match the frozen reference calls.
    """

    def __init__(self, max_steps: int = 12, snapshot_dir: str | Path | None = None) -> None:
        self.max_steps = max_steps
        self.rng = random.Random()
        self.task: dict[str, Any] = {}
        self.call_index = 0
        self.calls: list[dict[str, Any]] = []
        self.done = False
        self.success = False
        self.steps = 0
        self.off_path_calls = 0
        self._snapshots: dict[str, tuple[dict[str, Any], object]] = {}
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else None
        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, task: dict[str, Any], seed: int) -> dict[str, Any]:
        self.task = copy.deepcopy(task)
        self.rng.seed(seed)
        self.call_index = 0
        self.calls = []
        self.done = False
        self.success = False
        self.steps = 0
        self.off_path_calls = 0
        self._snapshots.clear()
        return self.observation()

    def observation(self) -> dict[str, Any]:
        return {
            "task_id": self.task.get("task_id"),
            "user_goal": self.task.get("user_goal"),
            "available_tools": self.task.get("available_tools", []),
            "call_index": self.call_index,
            "num_reference_actions": len(self.task.get("reference_actions", [])),
            "success_predicate_satisfied": self.call_index == len(self.task.get("reference_actions", [])),
            "off_path_calls": self.off_path_calls,
            "state_hash": self.state_hash(),
        }

    def expected_action(self) -> dict[str, Any] | None:
        references = self.task.get("reference_actions", [])
        return copy.deepcopy(references[self.call_index]) if self.call_index < len(references) else None

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode already terminated")
        self.steps += 1
        reward = 0.0
        reason = "running"
        tool_result: dict[str, Any] | None = None
        ground_truth_match = False
        if action.get("type") == "finish":
            self.done = True
            self.success = self.call_index == len(self.task.get("reference_actions", []))
            reward = float(self.success)
            ground_truth_match = self.success
            reason = "success" if self.success else "premature_finish"
        else:
            normalized = canonical_action(action)
            expected = self.expected_action()
            if expected is not None and normalized == canonical_action(expected):
                receipt_index = self.call_index
                ground_truth_match = True
                self.calls.append(normalized)
                self.call_index += 1
                receipts = self.task.get("reference_tool_receipts", [])
                if receipt_index < len(receipts):
                    tool_result = copy.deepcopy(receipts[receipt_index])
                    tool_result.update(
                        {"status": "ok", "call_index": self.call_index, "tool": normalized["tool"]}
                    )
                else:
                    tool_result = {"status": "ok", "call_index": self.call_index, "tool": normalized["tool"]}
                if self.call_index == len(self.task.get("reference_actions", [])):
                    self.done = True
                    self.success = True
                    reward = 1.0
                    reason = "success"
            elif self._is_executable(normalized):
                self.calls.append(normalized)
                self.off_path_calls += 1
                tool_result = {
                    "status": "no_effect",
                    "call_index": self.call_index,
                    "tool": normalized["tool"],
                    "reason": "valid_off_path_call",
                }
                reason = "recoverable_off_path_call"
            else:
                self.calls.append(normalized)
                self.done = True
                reason = "programmatic_checker_failure"
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            reason = "max_steps"
        return self.observation(), reward, self.done, {
            "terminal_reason": reason,
            "ground_truth_match": ground_truth_match,
            "tool_result": tool_result,
        }

    def snapshot(self) -> str:
        state = {
            "task": copy.deepcopy(self.task),
            "call_index": self.call_index,
            "calls": copy.deepcopy(self.calls),
            "done": self.done,
            "success": self.success,
            "steps": self.steps,
            "off_path_calls": self.off_path_calls,
            "rng_state": self.rng.getstate(),
        }
        payload = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        reference = f"apibank:{digest}" if self.snapshot_dir is not None else f"apibank:{uuid.uuid4().hex}"
        self._snapshots[reference] = (state, self.rng.getstate())
        if self.snapshot_dir is not None:
            (self.snapshot_dir / f"{digest}.json").write_text(payload + "\n", encoding="utf-8")
        return reference

    def restore(self, state_ref: str) -> None:
        if state_ref in self._snapshots:
            state, rng_state = self._snapshots[state_ref]
        elif self.snapshot_dir is not None and state_ref.startswith("apibank:"):
            path = self.snapshot_dir / f"{state_ref.split(':', 1)[1]}.json"
            state = json.loads(path.read_text(encoding="utf-8"))
            rng_state = self._tupleize(state.pop("rng_state"))
        else:
            raise KeyError(f"snapshot not found: {state_ref}")
        self.task = copy.deepcopy(state["task"])
        self.call_index = state["call_index"]
        self.calls = copy.deepcopy(state["calls"])
        self.done = state["done"]
        self.success = state["success"]
        self.steps = state["steps"]
        self.off_path_calls = state.get("off_path_calls", 0)
        self.rng.setstate(rng_state)

    @staticmethod
    def _tupleize(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(APIBankControlledEnv._tupleize(item) for item in value)
        if isinstance(value, dict):
            return {key: APIBankControlledEnv._tupleize(item) for key, item in value.items()}
        return value

    def get_rng_state(self) -> object:
        return self.rng.getstate()

    def set_rng_state(self, rng_state: object) -> None:
        self.rng.setstate(rng_state)

    def task_success(self) -> bool:
        return self.success

    def _is_executable(self, action: dict[str, Any]) -> bool:
        """Accept schema-valid off-path calls as recoverable no-ops.

        API-Bank traces do not expose a portable live backend state. The
        controlled environment therefore executes schema-valid non-reference
        calls as explicit no-ops: they consume a step but leave the success
        predicate unchanged. This creates real delayed-recovery trajectories
        without pretending that a hidden tool side effect occurred.
        """
        catalog = {str(tool.get("name", "")): tool for tool in self.task.get("available_tools", [])}
        schema = catalog.get(action["tool"])
        if schema is None:
            return False
        arguments = action.get("arguments", {})
        required = {str(name) for name in schema.get("required", [])}
        allowed = required | {str(name) for name in schema.get("optional", [])}
        return required <= set(arguments) and set(arguments) <= allowed

    def state_hash(self) -> str:
        payload = json.dumps(
            {
                "task_id": self.task.get("task_id"),
                "call_index": self.call_index,
                "calls": self.calls,
                "off_path_calls": self.off_path_calls,
                "done": self.done,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()
