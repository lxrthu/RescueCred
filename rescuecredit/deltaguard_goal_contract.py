from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Mapping, Sequence

from rescuecredit.deltaguard_observers import action_role
from rescuecredit.deltaguard_toolsandbox import public_structure_digest


GOAL_CONTRACT_VERSION = "pre-observation-goal-contract-v1"
UNKNOWN = "unknown"


_ROLE_FAMILY = {
    "get_cellular": "settings",
    "set_cellular": "settings",
    "get_wifi": "settings",
    "set_wifi": "settings",
    "get_location_service": "settings",
    "set_location_service": "settings",
    "get_low_battery": "settings",
    "set_low_battery": "settings",
    "send_message": "messaging",
    "search_messages": "messaging",
    "add_contact": "contacts",
    "modify_contact": "contacts",
    "remove_contact": "contacts",
    "search_contacts": "contacts",
    "add_reminder": "reminders",
    "modify_reminder": "reminders",
    "remove_reminder": "reminders",
    "search_reminder": "reminders",
}


def _function(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    function = schema.get("function", schema)
    return function if isinstance(function, Mapping) else {}


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def _digits(value: Any) -> str:
    return "".join(re.findall(r"\d", str(value)))


def _scalar_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        rows: list[Any] = []
        for key in sorted(value):
            rows.extend(_scalar_values(value[key]))
        return rows
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_scalar_values(item))
        return rows
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (str, int, float)):
        return [value]
    return []


def _value_kind(value: Any) -> str:
    digits = _digits(value)
    text = str(value)
    if len(digits) >= 7 and len(digits) >= len(re.sub(r"\D", "", text)):
        return "digits"
    return "text"


def _grounded_in_instruction(value: Any, instruction: str) -> bool:
    normalized = _normalized(value)
    if len(normalized) < 2:
        return False
    if normalized in _normalized(instruction):
        return True
    digits = _digits(value)
    return bool(len(digits) >= 7 and digits in _digits(instruction))


def _term_matches_value(term: Mapping[str, Any], value: Any) -> bool:
    if term.get("kind") == "digits":
        needle = str(term.get("normalized", ""))
        return bool(needle and needle in _digits(value))
    needle = str(term.get("normalized", ""))
    haystack = _normalized(value)
    return bool(needle and needle in haystack)


def infer_action_family(
    action: Mapping[str, Any], schemas: Sequence[Mapping[str, Any]]
) -> str | None:
    role = action_role(action, schemas)
    if role in _ROLE_FAMILY:
        return _ROLE_FAMILY[role]
    tool = str(action.get("tool", ""))
    schema = next(
        (
            _function(row)
            for row in schemas
            if str(_function(row).get("name", "")) == tool
        ),
        {},
    )
    text = _normalized(
        " ".join((tool, str(schema.get("description", ""))))
    )
    patterns = (
        ("messaging", ("message", "sms")),
        ("reminders", ("reminder",)),
        ("contacts", ("contact",)),
        ("settings", ("wifi", "cellular", "location service", "low battery")),
        ("time", ("timestamp", "datetime", "date", "holiday", "time zone")),
    )
    return next(
        (family for family, tokens in patterns if any(token in text for token in tokens)),
        None,
    )


def infer_goal_family(instruction: str) -> str | None:
    text = _normalized(instruction)
    patterns = (
        ("messaging", ("message", "text", "sms")),
        ("reminders", ("reminder", "remind")),
        ("contacts", ("contact", "phone number")),
        ("settings", ("wifi", "cellular", "location service", "low battery")),
        ("time", ("timestamp", "datetime", "date", "holiday", "time zone")),
    )
    matches = [
        family
        for family, tokens in patterns
        if any(re.search(rf"\b{re.escape(token)}\b", text) for token in tokens)
    ]
    return matches[0] if len(set(matches)) == 1 else None


def _contract_digest(contract: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(contract), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_goal_contract(
    *,
    instruction: str,
    action_a: Mapping[str, Any],
    action_b: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile public goal evidence before any branch is executed."""

    terms: dict[tuple[str, str], dict[str, Any]] = {}
    action_grounding: dict[str, list[str]] = {}
    action_families: dict[str, str | None] = {}
    for side, action in (("A", action_a), ("B", action_b)):
        grounded_keys = []
        for value in _scalar_values(action.get("arguments", {})):
            if not _grounded_in_instruction(value, instruction):
                continue
            kind = _value_kind(value)
            normalized = _digits(value) if kind == "digits" else _normalized(value)
            identity = (kind, normalized)
            term_id = "goal:" + hashlib.sha256(
                f"{kind}:{normalized}".encode("utf-8")
            ).hexdigest()[:16]
            terms[identity] = {
                "term_id": term_id,
                "kind": kind,
                "normalized": normalized,
                "provenance": "visible_instruction_and_fixed_action_argument",
            }
            grounded_keys.append(term_id)
        action_grounding[side] = sorted(set(grounded_keys))
        action_families[side] = infer_action_family(action, schemas)
    canonical_schemas = sorted(
        [dict(schema) for schema in schemas],
        key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True),
    )
    input_binding = {
        "instruction": instruction,
        "action_a": dict(action_a),
        "action_b": dict(action_b),
        "public_schemas": canonical_schemas,
    }
    contract = {
        "version": GOAL_CONTRACT_VERSION,
        "instruction_sha256": hashlib.sha256(
            instruction.encode("utf-8")
        ).hexdigest(),
        "goal_family": infer_goal_family(instruction),
        "input_sha256": _contract_digest(input_binding),
        "action_hashes": {
            "A": _contract_digest(dict(action_a)),
            "B": _contract_digest(dict(action_b)),
        },
        "action_structure_hashes": {
            "A": public_structure_digest(action_a),
            "B": public_structure_digest(action_b),
        },
        "public_schema_sha256": _contract_digest(
            {"schemas": canonical_schemas}
        ),
        "action_families": action_families,
        "action_grounded_term_ids": action_grounding,
        "terms": sorted(terms.values(), key=lambda row: str(row["term_id"])),
        "generated_before_receipts": True,
        "outcome_fields_read": [],
    }
    contract["sha256"] = _contract_digest(contract)
    return contract


def verify_goal_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("version") != GOAL_CONTRACT_VERSION:
        raise ValueError("unsupported Goal Contract version")
    if contract.get("generated_before_receipts") is not True:
        raise ValueError("Goal Contract is not marked pre-observation")
    if contract.get("outcome_fields_read") != []:
        raise ValueError("Goal Contract reports outcome access")
    supplied = contract.get("sha256")
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    if supplied != _contract_digest(payload):
        raise ValueError("Goal Contract digest mismatch")


def _receipt_term_count(
    receipt: Mapping[str, Any], terms: Sequence[Mapping[str, Any]]
) -> int | str:
    if receipt.get("exception"):
        return UNKNOWN
    values = _scalar_values(receipt.get("parsed"))
    matched_term_ids = {
        str(term["term_id"])
        for term in terms
        if any(_term_matches_value(term, value) for value in values)
    }
    return len(matched_term_ids)


def goal_predicate_rows(
    contract: Mapping[str, Any], evidence: Mapping[str, Any]
) -> list[dict[str, Any]]:
    verify_goal_contract(contract)
    rows: list[dict[str, Any]] = []
    goal_family = contract.get("goal_family")
    action_families = contract.get("action_families", {})
    if isinstance(goal_family, str):
        family_a = action_families.get("A")
        family_b = action_families.get("B")
        role_a = UNKNOWN if family_a is None else int(family_a == goal_family)
        role_b = UNKNOWN if family_b is None else int(family_b == goal_family)
        rows.append(
            {
                "predicate_id": "goal:role_alignment",
                "family": goal_family,
                "required": False,
                "routing_admissible": False,
                "delta_a": role_a,
                "delta_b": role_b,
                "evidence_scope": "visible instruction and public schema only",
            }
        )
    grounding = contract.get("action_grounded_term_ids", {})
    rows.append(
        {
            "predicate_id": "goal:grounded_argument_coverage",
            "family": str(goal_family or "goal"),
            "required": False,
            "routing_admissible": False,
            "delta_a": len(grounding.get("A", [])),
            "delta_b": len(grounding.get("B", [])),
            "evidence_scope": "instruction-grounded fixed action arguments",
        }
    )
    terms = contract.get("terms", [])
    if terms:
        receipt_a = evidence.get("branch_a", {}).get("action_receipt", {})
        receipt_b = evidence.get("branch_b", {}).get("action_receipt", {})
        rows.append(
            {
                "predicate_id": "goal:receipt_term_coverage",
                "family": str(goal_family or "goal"),
                "required": False,
                "routing_admissible": True,
                "delta_a": _receipt_term_count(receipt_a, terms),
                "delta_b": _receipt_term_count(receipt_b, terms),
                "evidence_scope": "public parsed action receipt",
            }
        )
    return rows
