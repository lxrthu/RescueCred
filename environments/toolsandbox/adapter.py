from __future__ import annotations

import copy
import hashlib
import json
import math
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# The only commit on apple/ToolSandbox at implementation time. Setup scripts
# verify this identity instead of silently following a moving default branch.
TOOL_SANDBOX_COMMIT = "165848b9a78cead7ca7fe7c89c688b58e6501219"


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(child) for child in value]
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        _plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _console_value_descriptor(value: Any) -> Any:
    """Return a deterministic descriptor for InteractiveConsole locals.

    Raw dill byte streams are reversible but not canonical: serializing two
    equivalent consoles can produce different bytes because memo/object identities
    differ. ToolSandbox's console contains imported modules/functions plus temporary
    JSON-like action values, so compare their executable names and stable values.
    """

    if isinstance(value, types.ModuleType):
        return {"kind": "module", "name": value.__name__}
    if callable(value):
        return {
            "kind": "callable",
            "module": getattr(value, "__module__", ""),
            "qualname": getattr(value, "__qualname__", getattr(value, "__name__", "")),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"kind": "value", "value": value}
    if isinstance(value, (list, tuple)):
        return {
            "kind": type(value).__name__,
            "items": [_console_value_descriptor(child) for child in value],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": {
                str(key): _console_value_descriptor(child)
                for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            },
        }
    return {
        "kind": "object",
        "type": type(value).__module__ + "." + type(value).__qualname__,
    }


def console_namespace_fingerprint(console: Any) -> Dict[str, Any]:
    namespace = getattr(console, "locals", None)
    if not isinstance(namespace, dict):
        raise TypeError("ToolSandbox InteractiveConsole.locals must be a dictionary")
    return {
        str(name): (
            {"kind": "python_builtins"}
            if str(name) == "__builtins__"
            else _console_value_descriptor(value)
        )
        for name, value in sorted(namespace.items(), key=lambda item: str(item[0]))
    }


def canonical_action(action: Mapping[str, Any]) -> Dict[str, Any]:
    tool = action.get("tool")
    arguments = action.get("arguments")
    if not isinstance(tool, str) or not tool:
        raise ValueError("action.tool must be a non-empty string")
    if not isinstance(arguments, Mapping):
        raise ValueError("action.arguments must be an object")
    return {
        "tool": tool,
        "arguments": {str(key): _plain(value) for key, value in sorted(arguments.items())},
    }


def schema_by_name(tool_schemas: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for entry in tool_schemas:
        function = entry.get("function", entry)
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            result[str(function["name"])] = dict(function)
    return result


def action_schema_complete(
    action: Mapping[str, Any], tool_schemas: Sequence[Mapping[str, Any]]
) -> bool:
    try:
        action = canonical_action(action)
    except (TypeError, ValueError):
        return False
    function = schema_by_name(tool_schemas).get(str(action["tool"]))
    if function is None:
        return False
    parameters = function.get("parameters", {})
    if not isinstance(parameters, Mapping):
        return False
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        return False
    arguments = action["arguments"]
    return set(required).issubset(arguments) and set(arguments).issubset(properties)


def controlled_missing_argument(
    action_b: Mapping[str, Any], tool_schemas: Sequence[Mapping[str, Any]]
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Create A by removing one public-schema required field from visible B.

    This is a controlled mechanism probe, not a naturally occurring Harness error.
    It never reads milestones, reference actions, evaluator values, or hidden state.
    """

    action_b = canonical_action(action_b)
    function = schema_by_name(tool_schemas).get(action_b["tool"])
    if function is None or not action_schema_complete(action_b, tool_schemas):
        return None
    parameters = function.get("parameters", {})
    required = parameters.get("required", []) if isinstance(parameters, Mapping) else []
    removable = sorted(
        field for field in required if field in action_b["arguments"]
    )
    if not removable:
        return None
    removed = removable[0]
    action_a = copy.deepcopy(action_b)
    del action_a["arguments"][removed]
    return action_a, removed


def score_decision(delta: float, atol: float = 1e-12) -> str:
    if delta > atol:
        return "rescue_preference"
    if delta < -atol:
        return "reverse_preference"
    return "zero_delta"


@dataclass
class ActionReceipt:
    action: Dict[str, Any]
    content: str
    exception: Optional[str]
    context: Any


class ToolSandboxRuntime:
    """Thin adapter over the pinned official ToolSandbox execution contract.

    Imports are delayed so the main RescueCredit environment can run unit tests
    without installing ToolSandbox's Python 3.9 dependency stack.
    """

    def __init__(self) -> None:
        try:
            from tool_sandbox.common.execution_context import (
                DatabaseNamespace,
                RoleType,
                ScenarioCategories,
                get_current_context,
                set_current_context,
            )
            from tool_sandbox.common.message_conversion import Message
            from tool_sandbox.common.tool_conversion import convert_to_openai_tools
            from tool_sandbox.common.tool_discovery import ToolBackend
            from tool_sandbox.roles.base_role import BaseRole
            from tool_sandbox.roles.execution_environment import ExecutionEnvironment
            from tool_sandbox.scenarios import named_scenarios
        except ImportError as error:
            raise RuntimeError(
                "ToolSandbox is not installed. Run scripts/cloud/"
                "setup_toolsandbox_stage0.sh in a separate Python 3.9 environment."
            ) from error
        self.DatabaseNamespace = DatabaseNamespace
        self.RoleType = RoleType
        self.ScenarioCategories = ScenarioCategories
        self.Message = Message
        self.BaseRole = BaseRole
        self.ExecutionEnvironment = ExecutionEnvironment
        self.convert_to_openai_tools = convert_to_openai_tools
        self.ToolBackend = ToolBackend
        self.named_scenarios = named_scenarios
        self.get_current_context = get_current_context
        self.set_current_context = set_current_context

    def scenarios(self) -> Dict[str, Any]:
        return self.named_scenarios(preferred_tool_backend=self.ToolBackend.DEFAULT)

    def select_scenarios(
        self,
        limit: int,
        seed: int,
        require_state_dependency: bool = False,
        require_multiple_tool: bool = True,
        require_single_user_turn: bool = True,
    ) -> List[Tuple[str, Any]]:
        import random

        state_dependency: List[Tuple[str, Any]] = []
        other_stateful: List[Tuple[str, Any]] = []
        for name, scenario in self.scenarios().items():
            categories = set(scenario.categories)
            if self.ScenarioCategories.NO_DISTRACTION_TOOLS not in categories:
                continue
            if require_state_dependency and self.ScenarioCategories.STATE_DEPENDENCY not in categories:
                continue
            if require_multiple_tool and self.ScenarioCategories.MULTIPLE_TOOL_CALL not in categories:
                continue
            if (
                require_single_user_turn
                and self.ScenarioCategories.SINGLE_USER_TURN not in categories
            ):
                continue
            context = scenario.starting_context
            tools = self._agent_tools(context)
            if any("rapid_api_search_tools" in tool.__module__ for tool in tools.values()):
                continue
            target = (
                state_dependency
                if self.ScenarioCategories.STATE_DEPENDENCY in categories
                else other_stateful
            )
            target.append((name, scenario))
        rng = random.Random(seed)
        rng.shuffle(state_dependency)
        rng.shuffle(other_stateful)
        return (state_dependency + other_stateful)[:limit]

    def _agent_tools(self, context: Any) -> Dict[str, Any]:
        """Apply the same visibility rule as official BaseRole.get_available_tools."""

        tools = context.get_available_tools(scrambling_allowed=True)
        return {
            name: tool
            for name, tool in tools.items()
            if self.RoleType.AGENT
            in getattr(tool, "visible_to", (self.RoleType.AGENT,))
        }

    def context_digest(self, context: Any) -> str:
        payload = context.to_dict(serialize_console=False)
        payload["interactive_console_namespace"] = console_namespace_fingerprint(
            context.interactive_console
        )
        return sha256_json(payload)

    def snapshot(self, context: Any) -> Any:
        return copy.deepcopy(context)

    def prepare(self, scenario: Any) -> Any:
        context = copy.deepcopy(scenario.starting_context)
        self.set_current_context(context)
        sandbox = context.get_database(
            self.DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )
        maximum = context.max_sandbox_message_index
        environment = self.ExecutionEnvironment()
        for index in range(maximum + 1):
            row = sandbox.filter(sandbox["sandbox_message_index"] == index)
            if row.is_empty():
                continue
            if (
                row["recipient"][0] == self.RoleType.EXECUTION_ENVIRONMENT
                and row["sender"][0] == self.RoleType.SYSTEM
            ):
                environment.respond(ending_index=index)
        if self.get_current_context().max_sandbox_message_index != maximum:
            raise RuntimeError("system initialization unexpectedly changed message history")
        return self.get_current_context()

    def tool_schemas(self, context: Any) -> List[Dict[str, Any]]:
        self.set_current_context(context)
        tools = self._agent_tools(context)
        return self.convert_to_openai_tools(tools)

    def visible_history(self, context: Any) -> List[Dict[str, Any]]:
        self.set_current_context(context)
        sandbox = context.get_database(
            self.DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )
        rows: List[Dict[str, Any]] = []
        for row in sandbox.to_dicts():
            visible_to = row.get("visible_to")
            if visible_to is not None and self.RoleType.AGENT not in visible_to:
                continue
            sender = str(row.get("sender"))
            recipient = str(row.get("recipient"))
            if sender == str(self.RoleType.SYSTEM):
                # The Harness may use the public behavioral instruction, but never
                # the execution-environment import bootstrap.
                if recipient == str(self.RoleType.EXECUTION_ENVIRONMENT):
                    continue
            rows.append(
                {
                    "sender": sender,
                    "recipient": recipient,
                    "content": str(row.get("content") or ""),
                    "tool_call_exception": row.get("tool_call_exception"),
                }
            )
        return rows

    def instruction(self, context: Any) -> str:
        messages = self.visible_history(context)
        user_messages = [
            row["content"]
            for row in messages
            if row["sender"] == str(self.RoleType.USER)
        ]
        return user_messages[-1] if user_messages else ""

    def execute(self, context: Any, action: Mapping[str, Any]) -> ActionReceipt:
        action = canonical_action(action)
        self.set_current_context(context)
        available = self._agent_tools(context)
        if action["tool"] not in available:
            raise ValueError("action selected an unavailable tool")
        execution_name = context.get_execution_facing_tool_name(action["tool"])
        arguments = json.dumps(
            action["arguments"], ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        code = (
            "_rescuecredit_arguments = json.loads(" + repr(arguments) + ")\n"
            "_rescuecredit_result = " + execution_name + "(**_rescuecredit_arguments)\n"
            "print(repr(_rescuecredit_result))"
        )
        self.BaseRole.add_messages(
            [
                self.Message(
                    sender=self.RoleType.AGENT,
                    recipient=self.RoleType.EXECUTION_ENVIRONMENT,
                    content=code,
                )
            ]
        )
        self.ExecutionEnvironment().respond()
        context = self.get_current_context()
        last = self.BaseRole.get_messages()[-1]
        return ActionReceipt(
            action=action,
            content=last.content,
            exception=last.tool_call_exception,
            context=context,
        )

    def official_score(self, scenario: Any, context: Any) -> Dict[str, Any]:
        result = scenario.evaluation.evaluate(
            execution_context=context, max_turn_count=scenario.max_messages
        )
        values = {
            "source": "official ToolSandbox EvaluationResult.similarity",
            "similarity": float(result.similarity),
            "milestone_similarity": float(result.milestone_similarity),
            "minefield_similarity": float(result.minefield_similarity),
            "turn_count": int(result.turn_count),
        }
        if not all(
            math.isfinite(float(values[key]))
            for key in (
                "similarity",
                "milestone_similarity",
                "minefield_similarity",
                "turn_count",
            )
        ):
            raise ValueError("official evaluator returned a non-finite value")
        return values
