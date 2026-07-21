from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from rescuecredit.deltaguard_observers import (
    action_role,
    public_role_map,
)
from rescuecredit.deltaguard_probe import canonical_action, parse_public_content


QUERY_VERSION = "goal-directed-query-v1"


@dataclass(frozen=True)
class GoalQuery:
    query_id: str
    family: str
    tool: str
    arguments: dict[str, Any]
    extractor: str
    expectation: str
    target_side: str = "B"
    hard_precondition: bool = True
    provenance: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["provenance"] = list(self.provenance)
        return row

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "GoalQuery":
        return cls(
            query_id=str(row["query_id"]),
            family=str(row["family"]),
            tool=str(row["tool"]),
            arguments=dict(row.get("arguments", {})),
            extractor=str(row["extractor"]),
            expectation=str(row["expectation"]),
            target_side=str(row.get("target_side", "B")),
            hard_precondition=bool(row.get("hard_precondition", True)),
            provenance=tuple(str(value) for value in row.get("provenance", ())),
        )


def _function(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    function = schema.get("function", schema)
    return function if isinstance(function, Mapping) else {}


def _schema_by_tool(
    schemas: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result = {}
    for schema in schemas:
        function = _function(schema)
        name = function.get("name")
        if isinstance(name, str):
            result[name] = function
    return result


def _type_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_type_matches(value, item) for item in expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "null":
        return value is None
    return True


def validate_action_schema(
    action: Mapping[str, Any], schemas: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Check only public JSON-schema obligations; no outcome or label access."""

    try:
        canonical = canonical_action(action)
    except ValueError as error:
        return {"valid": False, "violations": [str(error)]}
    function = _schema_by_tool(schemas).get(canonical["tool"])
    if function is None:
        return {"valid": False, "violations": ["tool_absent_from_public_schema"]}
    parameters = function.get("parameters", {})
    if not isinstance(parameters, Mapping):
        parameters = {}
    properties = parameters.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    required = parameters.get("required", [])
    if not isinstance(required, list):
        required = []
    arguments = canonical["arguments"]
    violations = []
    for key in required:
        if key not in arguments or arguments.get(key) is None:
            violations.append(f"missing_required:{key}")
    for key, value in arguments.items():
        specification = properties.get(key)
        if not isinstance(specification, Mapping):
            continue
        if value is None and key in required:
            continue
        expected_type = specification.get("type")
        if expected_type is not None and not _type_matches(value, expected_type):
            violations.append(f"type_mismatch:{key}:{expected_type}")
        enum = specification.get("enum")
        if isinstance(enum, list) and value not in enum:
            violations.append(f"enum_mismatch:{key}")
    return {"valid": not violations, "violations": violations}


def _allowed_arguments(
    tool: str,
    arguments: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    function = _schema_by_tool(schemas).get(tool, {})
    parameters = function.get("parameters", {})
    properties = parameters.get("properties", {}) if isinstance(parameters, Mapping) else {}
    if not isinstance(properties, Mapping):
        return {}
    return {
        key: value
        for key, value in arguments.items()
        if key in properties and value is not None
    }


def _query(
    *,
    family: str,
    tool: str,
    arguments: Mapping[str, Any],
    extractor: str,
    expectation: str,
    provenance: Sequence[str],
) -> GoalQuery:
    identity = {
        "family": family,
        "tool": tool,
        "arguments": dict(arguments),
        "extractor": extractor,
        "expectation": expectation,
        "target_side": "B",
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return GoalQuery(
        query_id=f"goal-query:{digest}",
        family=family,
        tool=tool,
        arguments=dict(arguments),
        extractor=extractor,
        expectation=expectation,
        provenance=tuple(provenance),
    )


def build_goal_directed_queries(
    *,
    action_a: Mapping[str, Any],
    action_b: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
    instruction: str,
) -> list[GoalQuery]:
    """Generate label-blind read-only tests of hard B preconditions.

    The first query is the frozen Pilot-0 choice. Candidates are ordered by
    specificity: referenced entity existence before broad service status.
    """

    del action_a, instruction
    roles = public_role_map(schemas)
    role_b = action_role(action_b, schemas)
    arguments = action_b.get("arguments", {})
    if not isinstance(arguments, Mapping):
        arguments = {}
    candidates: list[tuple[int, GoalQuery]] = []

    entity_specs = {
        "modify_contact": ("contacts", "search_contacts", "exists"),
        "remove_contact": ("contacts", "search_contacts", "exists"),
        "modify_reminder": ("reminders", "search_reminder", "exists"),
        "remove_reminder": ("reminders", "search_reminder", "exists"),
    }
    if role_b in entity_specs:
        family, query_role, expectation = entity_specs[role_b]
        tool = roles.get(query_role)
        if tool:
            preferred = (
                ("person_id", "phone_number", "name")
                if family == "contacts"
                else ("reminder_id", "content", "latitude", "longitude")
            )
            raw_query = {
                key: arguments[key]
                for key in preferred
                if key in arguments and arguments[key] is not None
            }
            filtered = _allowed_arguments(tool, raw_query, schemas)
            if filtered:
                candidates.append(
                    (
                        0,
                        _query(
                            family=family,
                            tool=tool,
                            arguments=filtered,
                            extractor="count",
                            expectation=expectation,
                            provenance=(
                                "B referenced entity and public read schema",
                                "hard existence precondition",
                            ),
                        ),
                    )
                )

    if role_b == "send_message" and roles.get("get_cellular"):
        candidates.append(
            (
                1,
                _query(
                    family="messaging",
                    tool=roles["get_cellular"],
                    arguments={},
                    extractor="boolean",
                    expectation="true",
                    provenance=("public dependency: sending requires cellular",),
                ),
            )
        )
    deduplicated = {query.query_id: (priority, query) for priority, query in candidates}
    return [
        query
        for _, query in sorted(
            deduplicated.values(), key=lambda item: (item[0], item[1].query_id)
        )
    ]


def query_structure(queries: Sequence[GoalQuery]) -> list[dict[str, Any]]:
    return [
        {
            "family": query.family,
            "tool": query.tool,
            "arguments": query.arguments,
            "extractor": query.extractor,
            "expectation": query.expectation,
            "target_side": query.target_side,
            "hard_precondition": query.hard_precondition,
        }
        for query in queries
    ]


def interpret_query_receipt(
    query: GoalQuery, receipt: Mapping[str, Any] | None
) -> dict[str, Any]:
    if receipt is None:
        return {"known": False, "supports_b": None, "value": None}
    if receipt.get("exception"):
        return {"known": False, "supports_b": None, "value": None}
    parsed = receipt.get("parsed")
    if query.extractor == "boolean":
        if not isinstance(parsed, bool):
            return {"known": False, "supports_b": None, "value": None}
        value: Any = parsed
    elif query.extractor == "count":
        if not isinstance(parsed, list):
            return {"known": False, "supports_b": None, "value": None}
        value = len(parsed)
    else:
        raise ValueError(f"unsupported query extractor: {query.extractor}")
    if query.expectation == "true":
        supports = value is True
    elif query.expectation == "exists":
        supports = isinstance(value, int) and value > 0
    elif query.expectation == "absent":
        supports = value == 0
    else:
        raise ValueError(f"unsupported query expectation: {query.expectation}")
    return {"known": True, "supports_b": supports, "value": value}


def build_goal_query_certificate(
    *,
    action_a: Mapping[str, Any],
    action_b: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
    query: GoalQuery | None,
    query_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    schema_a = validate_action_schema(action_a, schemas)
    schema_b = validate_action_schema(action_b, schemas)
    query_result = (
        interpret_query_receipt(query, query_receipt)
        if query is not None
        else {"known": False, "supports_b": None, "value": None}
    )
    schema_route = bool(schema_a["valid"] and not schema_b["valid"])
    query_route = bool(
        schema_a["valid"]
        and schema_b["valid"]
        and query is not None
        and query.hard_precondition
        and query_result["known"]
        and query_result["supports_b"] is False
    )
    reasons = []
    if schema_route:
        reasons.append("b_public_schema_violation")
    if query_route:
        reasons.append("b_hard_precondition_refuted")
    return {
        "version": QUERY_VERSION,
        "schema_a": schema_a,
        "schema_b": schema_b,
        "query": query.to_dict() if query is not None else None,
        "query_result": query_result,
        "schema_only_route_to_a": schema_route,
        "query_incremental_route_to_a": query_route,
        "route_to_a": bool(schema_route or query_route),
        "witness_reasons": reasons,
        "labels_read": False,
        "official_evaluator_called": False,
    }


def public_receipt_row(receipt: Any) -> dict[str, Any]:
    content = getattr(receipt, "content", None)
    exception = getattr(receipt, "exception", None)
    parsed = None if exception else parse_public_content(content)
    return {
        "action": canonical_action(getattr(receipt, "action")),
        "content": content,
        "exception": exception,
        "parsed": parsed,
    }
