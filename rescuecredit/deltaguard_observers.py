from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


REGISTRY_VERSION = "toolsandbox-public-observers-v1"


@dataclass(frozen=True)
class ObserverPredicate:
    predicate_id: str
    family: str
    tool: str
    arguments: dict[str, Any]
    extractor: str
    comparator: str
    target: Any = None
    required: bool = True
    provenance: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["provenance"] = list(self.provenance)
        return row

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "ObserverPredicate":
        return cls(
            predicate_id=str(row["predicate_id"]),
            family=str(row["family"]),
            tool=str(row["tool"]),
            arguments=dict(row.get("arguments", {})),
            extractor=str(row["extractor"]),
            comparator=str(row["comparator"]),
            target=row.get("target"),
            required=bool(row.get("required", True)),
            provenance=tuple(str(value) for value in row.get("provenance", ())),
        )


_ROLE_PATTERNS: dict[str, tuple[tuple[str, ...], ...]] = {
    "get_cellular": (("request", "cellular", "status"),),
    "set_cellular": (("enable", "disable", "cellular"),),
    "get_wifi": (("request", "wifi", "status"),),
    "set_wifi": (("enable", "disable", "wifi"),),
    "get_location_service": (("request", "location service", "status"),),
    "set_location_service": (("enable", "disable", "location service"),),
    "get_low_battery": (("request", "low battery", "status"),),
    "set_low_battery": (("enable", "disable", "low battery"),),
    "send_message": (("send", "message", "phone"),),
    "search_messages": (("search", "message"), ("matching", "messages")),
    "add_contact": (("add", "contact"),),
    "modify_contact": (("modify", "contact"),),
    "remove_contact": (("remove", "contact"),),
    "search_contacts": (("search", "contact"), ("matching", "contacts")),
    "add_reminder": (("add", "reminder"),),
    "modify_reminder": (("modify", "reminder"),),
    "remove_reminder": (("remove", "reminder"),),
    "search_reminder": (("search", "reminder"), ("matching", "reminders")),
}

_EXACT_ROLES = {
    "get_cellular_service_status": "get_cellular",
    "set_cellular_service_status": "set_cellular",
    "get_wifi_status": "get_wifi",
    "set_wifi_status": "set_wifi",
    "get_location_service_status": "get_location_service",
    "set_location_service_status": "set_location_service",
    "get_low_battery_mode_status": "get_low_battery",
    "set_low_battery_mode_status": "set_low_battery",
    "send_message_with_phone_number": "send_message",
    "search_messages": "search_messages",
    "add_contact": "add_contact",
    "modify_contact": "modify_contact",
    "remove_contact": "remove_contact",
    "search_contacts": "search_contacts",
    "add_reminder": "add_reminder",
    "modify_reminder": "modify_reminder",
    "remove_reminder": "remove_reminder",
    "search_reminder": "search_reminder",
}


def _function(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    function = schema.get("function", schema)
    return function if isinstance(function, Mapping) else {}


def schema_role(schema: Mapping[str, Any]) -> str | None:
    function = _function(schema)
    name = str(function.get("name", ""))
    if name in _EXACT_ROLES:
        return _EXACT_ROLES[name]
    text = " ".join(
        [name, str(function.get("description", ""))]
    ).casefold()
    for role, alternatives in _ROLE_PATTERNS.items():
        if any(all(token in text for token in pattern) for pattern in alternatives):
            return role
    return None


def public_role_map(
    schemas: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    ambiguous: set[str] = set()
    for schema in schemas:
        function = _function(schema)
        name = function.get("name")
        role = schema_role(schema)
        if not isinstance(name, str) or role is None:
            continue
        if role in result and result[role] != name:
            ambiguous.add(role)
        else:
            result[role] = name
    for role in ambiguous:
        result.pop(role, None)
    return result


def action_role(
    action: Mapping[str, Any], schemas: Sequence[Mapping[str, Any]]
) -> str | None:
    tool = str(action.get("tool", ""))
    for schema in schemas:
        if str(_function(schema).get("name", "")) == tool:
            return schema_role(schema)
    return _EXACT_ROLES.get(tool)


def _stable_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _predicate(
    *,
    family: str,
    tool: str,
    arguments: Mapping[str, Any],
    extractor: str,
    comparator: str,
    target: Any = None,
    required: bool = True,
    provenance: Sequence[str] = (),
) -> ObserverPredicate:
    identity = {
        "family": family,
        "tool": tool,
        "arguments": dict(arguments),
        "extractor": extractor,
        "comparator": comparator,
        "target": target,
    }
    return ObserverPredicate(
        predicate_id=f"{family}:{_stable_id(identity)}",
        family=family,
        tool=tool,
        arguments=dict(arguments),
        extractor=extractor,
        comparator=comparator,
        target=target,
        required=required,
        provenance=tuple(provenance),
    )


def _setting_target(
    role: str,
    role_actions: Sequence[tuple[str, Mapping[str, Any]]],
    instruction: str,
) -> bool | None:
    values = {
        bool(action.get("arguments", {}).get("on"))
        for action_role_name, action in role_actions
        if action_role_name == role
        and isinstance(action.get("arguments"), Mapping)
        and isinstance(action.get("arguments", {}).get("on"), bool)
    }
    if len(values) == 1:
        return next(iter(values))
    noun = {
        "set_cellular": "cellular",
        "set_wifi": "wifi",
        "set_location_service": "location service",
        "set_low_battery": "low battery",
    }[role]
    text = instruction.casefold()
    if noun not in text:
        return None
    if re.search(rf"(?:disable|turn\s+off|switch\s+off).{{0,24}}{re.escape(noun)}", text):
        return False
    if re.search(rf"(?:enable|turn\s+on|switch\s+on).{{0,24}}{re.escape(noun)}", text):
        return True
    return None


def _action_arguments(
    role_actions: Sequence[tuple[str, Mapping[str, Any]]], role: str
) -> list[tuple[str, Mapping[str, Any]]]:
    rows = []
    for side_role, action in role_actions:
        if side_role != role or not isinstance(action.get("arguments"), Mapping):
            continue
        rows.append((str(action.get("_side", "?")), action["arguments"]))
    return rows


def build_observer_plan(
    *,
    action_a: Mapping[str, Any],
    action_b: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
    instruction: str,
) -> list[ObserverPredicate]:
    """Build a contract-independent public observer plan.

    The plan is a deterministic union over immutable A/B and public schemas. It
    never reads branch outcomes, labels, evaluator state, or a generated contract.
    """

    roles = public_role_map(schemas)
    tagged_a = dict(action_a)
    tagged_b = dict(action_b)
    tagged_a["_side"] = "A"
    tagged_b["_side"] = "B"
    role_actions = [
        (action_role(action_a, schemas), tagged_a),
        (action_role(action_b, schemas), tagged_b),
    ]
    role_actions = [(str(role), action) for role, action in role_actions if role]
    active_roles = {role for role, _ in role_actions}
    predicates: list[ObserverPredicate] = []

    setting_pairs = (
        ("set_cellular", "get_cellular", "settings"),
        ("set_wifi", "get_wifi", "settings"),
        ("set_location_service", "get_location_service", "settings"),
        ("set_low_battery", "get_low_battery", "settings"),
    )
    if "send_message" in active_roles and "get_cellular" in roles:
        predicates.append(
            _predicate(
                family="messaging",
                tool=roles["get_cellular"],
                arguments={},
                extractor="boolean",
                comparator="target",
                target=True,
                provenance=("public dependency: messaging requires cellular",),
            )
        )
    for setter, getter, family in setting_pairs:
        if setter not in active_roles or getter not in roles:
            continue
        target = _setting_target(setter, role_actions, instruction)
        if target is None:
            continue
        predicates.append(
            _predicate(
                family=family,
                tool=roles[getter],
                arguments={},
                extractor="boolean",
                comparator="target",
                target=target,
                provenance=("immutable action arguments or visible instruction",),
            )
        )

    if "send_message" in active_roles and "search_messages" in roles:
        for side, arguments in _action_arguments(role_actions, "send_message"):
            phone = arguments.get("phone_number")
            content = arguments.get("content")
            if not isinstance(phone, str) or not isinstance(content, str):
                continue
            predicates.append(
                _predicate(
                    family="messaging",
                    tool=roles["search_messages"],
                    arguments={"recipient_phone_number": phone, "content": content},
                    extractor="count",
                    comparator="unique_add",
                    provenance=(f"immutable action {side}",),
                )
            )

    contact_roles = active_roles & {"add_contact", "modify_contact", "remove_contact"}
    if contact_roles and "search_contacts" in roles:
        for role in sorted(contact_roles):
            for side, arguments in _action_arguments(role_actions, role):
                query: dict[str, Any] = {}
                for key in ("person_id", "phone_number", "name", "is_self"):
                    if key in arguments:
                        query[key] = arguments[key]
                        break
                if not query:
                    continue
                comparator = (
                    "decrease"
                    if role == "remove_contact"
                    else "unique_add"
                    if role == "add_contact"
                    else "increase"
                )
                target = None
                extractor = "count"
                if role == "modify_contact":
                    target = {
                        key: value
                        for key, value in arguments.items()
                        if key != "person_id"
                    }
                    extractor = "match_count"
                predicates.append(
                    _predicate(
                        family="contacts",
                        tool=roles["search_contacts"],
                        arguments=query,
                        extractor=extractor,
                        comparator=comparator,
                        target=target,
                        provenance=(f"immutable action {side}",),
                    )
                )

    reminder_roles = active_roles & {"add_reminder", "modify_reminder", "remove_reminder"}
    if reminder_roles and "search_reminder" in roles:
        for role in sorted(reminder_roles):
            for side, arguments in _action_arguments(role_actions, role):
                query: dict[str, Any] = {}
                for key in ("reminder_id", "content", "latitude", "longitude"):
                    if key in arguments:
                        query[key] = arguments[key]
                        break
                if not query:
                    continue
                comparator = (
                    "decrease"
                    if role == "remove_reminder"
                    else "unique_add"
                    if role == "add_reminder"
                    else "increase"
                )
                target = None
                extractor = "count"
                if role == "modify_reminder":
                    target = {
                        key: value
                        for key, value in arguments.items()
                        if key != "reminder_id"
                    }
                    extractor = "match_count"
                predicates.append(
                    _predicate(
                        family="reminders",
                        tool=roles["search_reminder"],
                        arguments=query,
                        extractor=extractor,
                        comparator=comparator,
                        target=target,
                        provenance=(f"immutable action {side}",),
                    )
                )

    deduplicated = {predicate.predicate_id: predicate for predicate in predicates}
    return [deduplicated[key] for key in sorted(deduplicated)]


def plan_family(plan: Sequence[ObserverPredicate]) -> str | None:
    families = sorted({predicate.family for predicate in plan})
    if not families:
        return None
    if "messaging" in families:
        return "messaging"
    return families[0] if len(families) == 1 else "+".join(families)


def plan_digest(plan: Sequence[ObserverPredicate]) -> str:
    payload = [predicate.to_dict() for predicate in plan]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def plan_structure_payload(
    plan: Sequence[ObserverPredicate],
) -> list[dict[str, Any]]:
    rows = [
        {
            key: value
            for key, value in predicate.to_dict().items()
            if key != "predicate_id"
        }
        for predicate in plan
    ]
    return sorted(
        rows,
        key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, default=str),
    )


def validate_public_plan(
    plan: Sequence[ObserverPredicate], schemas: Sequence[Mapping[str, Any]]
) -> None:
    role_map = public_role_map(schemas)
    allowed = set(role_map.values())
    for predicate in plan:
        if predicate.tool not in allowed:
            raise ValueError(f"observer is not a whitelisted public read tool: {predicate.tool}")
        role = next((role for role, name in role_map.items() if name == predicate.tool), None)
        if role not in {
            "get_cellular",
            "get_wifi",
            "get_location_service",
            "get_low_battery",
            "search_messages",
            "search_contacts",
            "search_reminder",
        }:
            raise ValueError(f"observer role is not read-only: {role}")
