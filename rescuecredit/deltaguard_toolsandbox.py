from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from rescuecredit.deltaguard_probe import canonical_action, parse_public_content


_UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


def _parse_execution_call(code: str) -> tuple[str, dict[str, Any]] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    arguments: dict[str, Any] | None = None
    execution_name: str | None = None
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(statement.value, ast.Call):
            continue
        if target.id == "_rescuecredit_arguments" and statement.value.args:
            try:
                encoded = ast.literal_eval(statement.value.args[0])
                decoded = json.loads(encoded)
            except (SyntaxError, ValueError, TypeError):
                continue
            if isinstance(decoded, dict):
                arguments = decoded
        elif target.id == "_rescuecredit_result":
            execution_name = ast.unparse(statement.value.func)
    if arguments is None or execution_name is None:
        return None
    return execution_name, arguments


def _replace(value: Any, replacements: Mapping[Any, Any]) -> Any:
    if isinstance(value, Mapping):
        return {key: _replace(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace(child, replacements) for child in value]
    if isinstance(value, tuple):
        return tuple(_replace(child, replacements) for child in value)
    return replacements.get(value, value)


def _replace_text(value: str, replacements: Mapping[Any, Any]) -> str:
    result = value
    for old, new in replacements.items():
        result = result.replace(str(old), str(new))
    return result


def _structural(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _structural(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_structural(child) for child in value]
    if isinstance(value, str):
        return _UUID.sub("<PUBLIC_ID>", value)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 1_000_000_000:
        return "<PUBLIC_TIME>"
    return value


def public_structure_digest(value: Any) -> str:
    payload = json.dumps(_structural(value), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_visible_row(
    row: Mapping[str, Any], replacements: Mapping[Any, Any]
) -> dict[str, Any]:
    content = _replace_text(str(row.get("content", "")), replacements)
    parsed_call = _parse_execution_call(content)
    if parsed_call is not None:
        execution_name, arguments = parsed_call
        normalized_content: Any = {
            "execution_name": execution_name,
            "arguments": _structural(arguments),
        }
    else:
        parsed = parse_public_content(content)
        normalized_content = _structural(parsed)
    exception = row.get("tool_call_exception")
    return {
        "sender": str(row.get("sender", "")),
        "recipient": str(row.get("recipient", "")),
        "content": normalized_content,
        "tool_call_exception": (
            _structural(_replace_text(str(exception), replacements))
            if exception is not None
            else None
        ),
    }


def normalized_visible_history_digest(
    rows: Sequence[Mapping[str, Any]], replacements: Mapping[Any, Any]
) -> str:
    normalized = [_normalized_visible_row(row, replacements) for row in rows]
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collect_replacements(old: Any, new: Any, result: dict[Any, Any]) -> None:
    if isinstance(old, str) and isinstance(new, str) and old != new and _UUID.match(old):
        result[old] = new
        return
    if (
        isinstance(old, (int, float))
        and not isinstance(old, bool)
        and isinstance(new, (int, float))
        and not isinstance(new, bool)
        and old != new
        and float(old) > 1_000_000_000
    ):
        result[old] = new
        return
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        for key in old.keys() & new.keys():
            _collect_replacements(old[key], new[key], result)
    elif isinstance(old, Sequence) and not isinstance(old, (str, bytes)) and isinstance(
        new, Sequence
    ) and not isinstance(new, (str, bytes)):
        for left, right in zip(old, new):
            _collect_replacements(left, right, result)


def _public_tool_for_execution(runtime: Any, context: Any, execution_name: str) -> str:
    matches = []
    for schema in runtime.tool_schemas(context):
        function = schema.get("function", {})
        public_name = function.get("name")
        if not isinstance(public_name, str):
            continue
        if context.get_execution_facing_tool_name(public_name) == execution_name:
            matches.append(public_name)
    if len(matches) != 1:
        raise ValueError("visible execution tool is unavailable or ambiguous")
    return matches[0]


def replay_visible_prefix(
    *, runtime: Any, scenario: Any, target_history: Sequence[Mapping[str, Any]]
) -> tuple[Any, dict[Any, Any], dict[str, int]]:
    """Replay only visible tool calls, remapping nondeterministic public IDs.

    Unlike V8, this deliberately does not require receipt text equality. UUIDs
    and creation timestamps are public but nondeterministic; their replacements
    are propagated into later visible actions and the frozen A/B arguments.
    """

    prefix = runtime.prepare(scenario)
    replacements: dict[Any, Any] = {}
    replayed = 0
    remapped_values = 0
    for index, row in enumerate(target_history):
        parsed = _parse_execution_call(str(row.get("content", "")))
        if parsed is None:
            continue
        execution_name, old_arguments = parsed
        public_name = _public_tool_for_execution(runtime, prefix, execution_name)
        arguments = _replace(old_arguments, replacements)
        receipt = runtime.execute(prefix, {"tool": public_name, "arguments": arguments})
        prefix = receipt.context
        replayed += 1
        old_result = None
        for later in target_history[index + 1 :]:
            if _parse_execution_call(str(later.get("content", ""))) is not None:
                break
            if str(later.get("sender", "")).endswith("EXECUTION_ENVIRONMENT"):
                old_result = parse_public_content(later.get("content"))
                break
        before = len(replacements)
        if old_result is not None and receipt.exception is None:
            _collect_replacements(
                old_result, parse_public_content(receipt.content), replacements
            )
        remapped_values += len(replacements) - before
    target_digest = normalized_visible_history_digest(target_history, replacements)
    replayed_history = runtime.visible_history(prefix)
    replay_digest = normalized_visible_history_digest(replayed_history, {})
    if target_digest != replay_digest:
        raise RuntimeError("normalized public prefix replay diverged from frozen history")
    return prefix, replacements, {
        "prefix_actions_replayed": replayed,
        "nondeterministic_values_remapped": remapped_values,
        "normalized_visible_history_equal": True,
        "normalized_visible_history_sha256": replay_digest,
    }


def remap_frozen_action(
    action: Mapping[str, Any], replacements: Mapping[Any, Any]
) -> dict[str, Any]:
    return canonical_action(_replace(action, replacements))
