from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    for method in ("model_dump", "dict", "to_dict"):
        candidate = getattr(value, method, None)
        if callable(candidate):
            return _plain(candidate())
    return str(value)


def normalize_function_tools(api_docs: Any) -> list[dict[str, Any]]:
    """Convert public AppWorld function docs to RescueCredit's action schema."""

    if hasattr(api_docs, "function_calling"):
        api_docs = api_docs.function_calling()
    payload = _plain(api_docs)
    def entries(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for child in value for item in entries(child)]
        if not isinstance(value, dict):
            return []
        if "function" in value or ("name" in value and "parameters" in value):
            return [value]
        return [item for child in value.values() for item in entries(child)]

    payload = entries(payload)
    if not payload:
        raise TypeError("AppWorld function-calling docs contain no function entries")

    normalized: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function", entry)
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", ""))
        if not name:
            continue
        parameters = function.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        required = [str(item) for item in parameters.get("required", [])]
        normalized.append(
            {
                "name": name,
                "required": required,
                "optional": sorted(set(map(str, properties)) - set(required)),
                "parameter_schema": _plain(properties),
                "description": str(function.get("description", "")),
            }
        )
    return sorted(normalized, key=lambda item: item["name"])


def _split_tool_name(action: dict[str, Any]) -> tuple[str, str]:
    app = str(action.get("app", ""))
    api = str(action.get("api", ""))
    if not app or not api:
        name = str(action.get("tool", ""))
        for separator in ("__", "."):
            if separator in name:
                app, api = name.split(separator, 1)
                break
    if not _IDENTIFIER.fullmatch(app) or not _IDENTIFIER.fullmatch(api):
        raise ValueError(f"invalid AppWorld tool identifier: {app!r}.{api!r}")
    return app, api


def canonical_appworld_action(action: dict[str, Any]) -> dict[str, Any]:
    app, api = _split_tool_name(action)
    return {
        "tool": f"{app}__{api}",
        "arguments": {
            str(key): value
            for key, value in sorted(dict(action.get("arguments", {})).items())
        },
    }


def render_atomic_call(action: dict[str, Any]) -> str:
    """Render one identifier-checked call whose arguments cross as JSON."""

    app, api = _split_tool_name(action)
    raw_arguments = action.get("arguments", {})
    if not isinstance(raw_arguments, dict):
        raise TypeError("AppWorld action arguments must be a JSON object")
    arguments = json.dumps(
        raw_arguments,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    if len(arguments.encode("utf-8")) > 1_000_000:
        raise ValueError("AppWorld action arguments exceed the 1 MB safety limit")
    return (
        "import json\n"
        "print(json.dumps("
        f"apis.{app}.{api}(**json.loads({arguments!r})), "
        "ensure_ascii=False, sort_keys=True, default=str))"
    )


def _extract_success(result: Any) -> bool:
    if isinstance(result, bool):
        return result
    if isinstance(result, (int, float)):
        return float(result) > 0.0
    if isinstance(result, dict):
        for key in (
            "success",
            "passed",
            "task_success",
            "task_goal_completion",
            "task_goal_completion_rate",
        ):
            if key in result:
                return _extract_success(result[key])
    for key in ("success", "passed", "task_success"):
        if hasattr(result, key):
            return _extract_success(getattr(result, key))
    raise TypeError(f"unsupported AppWorld evaluation result: {type(result).__name__}")


@dataclass
class _Snapshot:
    appworld_state_id: str
    db_state_hash: str
    rng_state: object
    global_random_state: object
    numpy_random_state: object | None
    world_control_state: dict[str, Any]
    history: list[dict[str, Any]]
    steps: int
    done: bool


class AppWorldAtomicEnv:
    """Reference-free atomic function-call wrapper for official AppWorld."""

    def __init__(
        self,
        experiment_name: str = "rescuecredit_appworld",
        max_steps: int = 40,
        world_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.max_steps = int(max_steps)
        self._world_factory = world_factory
        self.world: Any | None = None
        self.rng = random.Random()
        self.task_id = ""
        self.history: list[dict[str, Any]] = []
        self.steps = 0
        self.done = False
        self._snapshots: dict[str, _Snapshot] = {}
        self._tools: list[dict[str, Any]] = []
        self._db_state_hash: str | None = None

    def _factory(self) -> Callable[..., Any]:
        if self._world_factory is not None:
            return self._world_factory
        try:
            from appworld import AppWorld
        except ImportError as error:
            raise RuntimeError(
                "Install AppWorld with `pip install appworld`, then run "
                "`appworld install` and `appworld download data`."
            ) from error
        return AppWorld

    def reset(self, task: dict[str, Any], seed: int) -> dict[str, Any]:
        self.close()
        self.task_id = str(task["task_id"])
        self.max_steps = int(task.get("max_steps", self.max_steps))
        self.rng.seed(seed)
        self.history = []
        self.steps = 0
        self.done = False
        self._snapshots.clear()
        self.world = self._factory()(
            task_id=self.task_id,
            experiment_name=self.experiment_name,
            ground_truth_mode="minimal",
            raise_on_failure=False,
            random_seed=seed,
        )
        self._tools = normalize_function_tools(self.world.task.api_docs)
        initial_state_id = str(self.world.save_state())
        self._db_state_hash = self._checkpoint_digest(initial_state_id)
        return self.observation()

    def observation(self) -> dict[str, Any]:
        if self.world is None:
            raise RuntimeError("environment is not reset")
        return {
            "task_id": self.task_id,
            "user_goal": str(self.world.task.instruction),
            "available_tools": copy.deepcopy(self._tools),
            "history": copy.deepcopy(self.history),
            "call_index": self.steps,
            "goal_satisfied": bool(self.done),
            "state_hash": self.state_hash(),
            "reference_free_observation": True,
        }

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.world is None:
            raise RuntimeError("environment is not reset")
        if self.done:
            raise RuntimeError("episode already terminated")
        normalized = canonical_appworld_action(action)
        output = self.world.execute(render_atomic_call(normalized))
        output_text = str(output)
        lowered = output_text.lower()
        failed = any(
            marker in lowered
            for marker in ("execution failed", "traceback", "response status code is")
        )
        receipt = {
            "status": "error" if failed else "ok",
            "tool": normalized["tool"],
            "output": output_text,
        }
        self.steps += 1
        self.history.append({"action": normalized, "tool_result": receipt})
        live_state_id = str(self.world.save_state())
        self._db_state_hash = self._checkpoint_digest(live_state_id)
        completed = getattr(self.world, "task_completed", lambda: False)()
        self.done = bool(completed or self.steps >= self.max_steps)
        terminal_reason = (
            "task_claimed_complete"
            if completed
            else "max_steps"
            if self.done
            else "tool_error"
            if failed
            else "running"
        )
        return self.observation(), 0.0, self.done, {
            "terminal_reason": terminal_reason,
            "tool_result": receipt,
            "execution_error": failed,
        }

    def snapshot(self) -> str:
        if self.world is None:
            raise RuntimeError("environment is not reset")
        appworld_state_id = str(self.world.save_state())
        db_state_hash = self._checkpoint_digest(appworld_state_id)
        if db_state_hash is None:
            raise RuntimeError(
                "cannot locate a deterministic AppWorld checkpoint export; "
                "Shadow replay is disabled until the DB digest contract is known"
            )
        numpy_random_state = None
        try:
            import numpy as np

            numpy_random_state = copy.deepcopy(np.random.get_state())
        except ImportError:
            pass
        world_control_state = {
            name: copy.deepcopy(getattr(self.world, name))
            for name in ("environment_io", "num_interactions", "num_sub_interactions")
            if hasattr(self.world, name)
        }
        reference = f"appworld:{uuid.uuid4().hex}"
        self._snapshots[reference] = _Snapshot(
            appworld_state_id=appworld_state_id,
            db_state_hash=db_state_hash,
            rng_state=self.rng.getstate(),
            global_random_state=random.getstate(),
            numpy_random_state=numpy_random_state,
            world_control_state=world_control_state,
            history=copy.deepcopy(self.history),
            steps=self.steps,
            done=self.done,
        )
        return reference

    def restore(self, state_ref: str) -> None:
        if self.world is None:
            raise RuntimeError("environment is not reset")
        snapshot = self._snapshots.get(state_ref)
        if snapshot is None:
            raise KeyError(f"snapshot not found: {state_ref}")
        self.world.load_state(snapshot.appworld_state_id)
        for name, value in snapshot.world_control_state.items():
            setattr(self.world, name, copy.deepcopy(value))
        self.rng.setstate(snapshot.rng_state)
        random.setstate(snapshot.global_random_state)
        if snapshot.numpy_random_state is not None:
            import numpy as np

            np.random.set_state(snapshot.numpy_random_state)
        self.history = copy.deepcopy(snapshot.history)
        self.steps = snapshot.steps
        self.done = snapshot.done
        restored_state_id = str(self.world.save_state())
        restored_hash = self._checkpoint_digest(restored_state_id)
        if restored_hash != snapshot.db_state_hash:
            raise RuntimeError(
                "AppWorld DB state digest differs after restore; causal Shadow is invalid"
            )
        self._db_state_hash = restored_hash

    def get_rng_state(self) -> object:
        return self.rng.getstate()

    def set_rng_state(self, rng_state: object) -> None:
        self.rng.setstate(rng_state)

    def evaluate(self) -> bool:
        if self.world is None:
            raise RuntimeError("environment is not reset")
        save = getattr(self.world, "save", None)
        if callable(save):
            save()
        result = self.world.evaluate()
        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            result = to_dict()
        return _extract_success(result)

    def task_success(self) -> bool:
        return self.evaluate()

    def state_hash(self) -> str:
        payload = json.dumps(
            {
                "task_id": self.task_id,
                "history": self.history,
                "steps": self.steps,
                "done": self.done,
                "db_state_hash": self._db_state_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _hash_path(path: Path) -> str:
        digest = hashlib.sha256()
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
            return digest.hexdigest()
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode())
            digest.update(child.read_bytes())
        return digest.hexdigest()

    def _checkpoint_digest(self, state_id: str) -> str | None:
        if self.world is None:
            return None
        export_state = getattr(self.world, "export_state", None)
        if callable(export_state):
            payload = json.dumps(
                export_state(), sort_keys=True, default=str, separators=(",", ":")
            )
            return hashlib.sha256(payload.encode()).hexdigest()
        state_path = Path(state_id)
        if state_path.exists():
            return self._hash_path(state_path)
        roots = []
        for name in ("output_directory", "output_dir"):
            value = getattr(self.world, name, None)
            if value:
                roots.append(Path(value))
        for root in roots:
            candidates = [root / "checkpoints" / state_id, root / state_id]
            for candidate in candidates:
                if candidate.exists():
                    return self._hash_path(candidate)
        return None

    def close(self) -> None:
        if self.world is not None:
            close = getattr(self.world, "close", None)
            if callable(close):
                close()
        self.world = None

    def __enter__(self) -> "AppWorldAtomicEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
