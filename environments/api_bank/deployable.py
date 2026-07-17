from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from rescuecredit.types import HarnessDecision

from .adapter import canonical_action

SemanticLabel = Literal["true", "false", "unknown"]
CorrectionGenerator = Callable[
    [dict[str, Any], dict[str, Any], str, dict[str, Any] | None],
    dict[str, Any] | None,
]


@dataclass(frozen=True)
class ActionValidity:
    executable_valid: bool
    semantic_valid: SemanticLabel
    reason: str
    evidence: tuple[str, ...] = ()


def public_harness_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Return only information available at intervention time.

    In particular, reference actions, expected actions, target signatures and
    dataset labels are never copied into this view.
    """

    return {
        "user_goal": observation.get("user_goal", ""),
        "available_tools": copy.deepcopy(observation.get("available_tools", [])),
        "call_index": observation.get("call_index", 0),
        "goal_satisfied": bool(
            observation.get("goal_satisfied", observation.get("success_predicate_satisfied", False))
        ),
        "state_hash": observation.get("state_hash"),
    }


def _normalize(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9@._:+-]+", str(value).lower()))


def _flatten_receipt_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        flattened: list[str] = []
        for child in value.values():
            flattened.extend(_flatten_receipt_values(child))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = []
        for child in value:
            flattened.extend(_flatten_receipt_values(child))
        return flattened
    if value is None or isinstance(value, bool):
        return []
    normalized = _normalize(value)
    return [normalized] if normalized else []


def merge_visible_tool_context(
    context: dict[str, Any] | None,
    latest_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Retain successful visible receipts across later failed/no-op calls.

    The Harness may use only values that have already been returned to the
    policy.  A later off-path call must not erase an earlier authentication
    token or other successful tool output.
    """

    merged = copy.deepcopy(context or {})
    if not latest_result:
        return merged or None
    latest = copy.deepcopy(latest_result)
    if latest.get("status") == "ok":
        merged.update(latest)
        merged.pop("latest_feedback", None)
        return merged
    if not merged:
        return latest
    merged["latest_feedback"] = latest
    if "suggested_action" in latest:
        merged["suggested_action"] = copy.deepcopy(latest["suggested_action"])
    return merged


class VisibleContextSemanticValidator:
    """Conservative tri-state validator using only visible context.

    Schema checks can prove invalidity.  Semantic validity is returned as
    ``true`` only when every argument is supported by the current user goal or
    an already observed tool receipt.  Everything else is ``unknown`` rather
    than silently treating executability as semantic correctness.
    """

    def explicit_value(
        self,
        observation: dict[str, Any],
        parameter: str,
        previous_tool_result: dict[str, Any] | None = None,
    ) -> Any | None:
        values = self.explicit_values(observation, parameter, previous_tool_result)
        return values[0] if len(values) == 1 else None

    def explicit_values(
        self,
        observation: dict[str, Any],
        parameter: str,
        previous_tool_result: dict[str, Any] | None = None,
    ) -> list[Any]:
        goal = str(observation.get("user_goal", ""))
        label = re.escape(parameter.replace("_", " "))
        value_pattern = (
            r"(?:is|equals|=|:)\s*"
            r"(?:\"([^\"]+)\"|'([^']+)'|(.+?)(?=\s+(?:and|but)\b|[\n,;.]|$))"
        )
        values: list[Any] = []
        for match in re.finditer(rf"\b(?:my\s+)?{label}\b\s*{value_pattern}", goal, flags=re.IGNORECASE):
            value = next((group for group in match.groups() if group is not None), "").strip()
            if value and _normalize(value) not in {_normalize(existing) for existing in values}:
                values.append(value)
        if previous_tool_result:
            for key, value in previous_tool_result.items():
                if str(key).lower().replace("_", " ") == parameter.lower().replace("_", " "):
                    if _normalize(value) not in {_normalize(existing) for existing in values}:
                        values.append(value)
        normalized_parameter = parameter.lower().replace("_", " ").strip()
        if not values and normalized_parameter.endswith(" id"):
            identifiers = []
            for candidate in re.findall(r"(?<!\d)\d{6,}(?!\d)", goal):
                if candidate not in identifiers:
                    identifiers.append(candidate)
            if len(identifiers) == 1:
                values.append(identifiers[0])
        return values

    def prerequisite_input_value(
        self,
        observation: dict[str, Any],
        parameter: str,
        previous_tool_result: dict[str, Any] | None = None,
    ) -> Any | None:
        """Resolve a producer input without confusing old/new credentials.

        A consumer such as ``ModifyPassword`` may make the generic label
        ``password`` ambiguous because both an old and a new password are
        visible.  Authentication producers need the current credential, so a
        uniquely visible ``old_password`` or ``current_password`` is safe to
        use.  This rule consumes only intervention-time user text/receipts.
        """

        value = self.explicit_value(observation, parameter, previous_tool_result)
        if value is not None:
            return value
        aliases = {
            "password": ("old_password", "current_password"),
        }
        candidates: list[Any] = []
        for alias in aliases.get(parameter.lower(), ()):
            for candidate in self.explicit_values(observation, alias, previous_tool_result):
                if _normalize(candidate) not in {_normalize(item) for item in candidates}:
                    candidates.append(candidate)
        return candidates[0] if len(candidates) == 1 else None

    def validate(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        previous_tool_result: dict[str, Any] | None = None,
    ) -> ActionValidity:
        observation = public_harness_observation(observation)
        if action.get("type") == "finish":
            valid = bool(observation.get("goal_satisfied"))
            return ActionValidity(valid, "true" if valid else "false", "visible goal-satisfaction predicate")

        tool_name = str(action.get("tool", ""))
        arguments = dict(action.get("arguments", {}))
        catalog = {str(entry.get("name", "")): entry for entry in observation.get("available_tools", [])}
        schema = catalog.get(tool_name)
        if schema is None:
            return ActionValidity(False, "false", "tool is absent from the visible action schema")
        required = {str(name) for name in schema.get("required", [])}
        optional = {str(name) for name in schema.get("optional", [])}
        missing = sorted(required - set(arguments))
        unexpected = sorted(set(arguments) - required - optional)
        if missing:
            return ActionValidity(False, "false", f"missing required arguments: {missing}")
        if unexpected:
            return ActionValidity(False, "false", f"unexpected arguments: {unexpected}")

        context_values = {_normalize(observation.get("user_goal", ""))}
        context_values.update(_flatten_receipt_values(previous_tool_result))
        evidence: list[str] = []
        unknown: list[str] = []
        contradictions: list[str] = []
        for name, value in arguments.items():
            normalized = _normalize(value)
            explicit = self.explicit_value(observation, str(name), previous_tool_result)
            if explicit is not None:
                if _normalize(explicit) == normalized:
                    evidence.append(f"{name}:explicit")
                else:
                    contradictions.append(str(name))
                continue
            if normalized and any(normalized in context for context in context_values if context):
                evidence.append(f"{name}:visible_context")
            else:
                unknown.append(str(name))
        if contradictions:
            return ActionValidity(True, "false", f"arguments contradict visible values: {contradictions}", tuple(evidence))
        if unknown:
            return ActionValidity(True, "unknown", f"semantic support unavailable for: {unknown}", tuple(evidence))
        return ActionValidity(True, "true", "all action arguments are supported by visible context", tuple(evidence))

    def locally_supported_for_action(
        self,
        observation: dict[str, Any],
        value: Any,
        existing_arguments: dict[str, Any],
        max_distance: int = 160,
    ) -> bool:
        """Require a goal-derived repair value to be near another action value.

        This conservative association prevents a value from a later intent in
        a multi-intent goal from being copied into the current tool call.
        """

        goal = _normalize(observation.get("user_goal", ""))
        candidate = _normalize(value)
        anchors = [_normalize(item) for item in existing_arguments.values() if _normalize(item)]
        if not goal or not candidate or not anchors:
            return False
        candidate_positions = [match.start() for match in re.finditer(re.escape(candidate), goal)]
        anchor_positions = [
            match.start()
            for anchor in anchors
            for match in re.finditer(re.escape(anchor), goal)
        ]
        return bool(
            candidate_positions
            and anchor_positions
            and min(abs(candidate_pos - anchor_pos) for candidate_pos in candidate_positions for anchor_pos in anchor_positions)
            <= max_distance
        )


class DeployableAPIBankHarness:
    """Narrow reference-free harness for paper-facing experiments.

    Its API deliberately has no ``expected`` or ``reference_actions`` input.
    It only applies a correction when the corrected action is semantically
    provable from information visible at intervention time.
    """

    CONDITIONS = {"H0", "H1", "H2", "H3", "Hplacebo"}

    def __init__(
        self,
        condition: str = "H3",
        validator: VisibleContextSemanticValidator | None = None,
        correction_generator: CorrectionGenerator | None = None,
    ) -> None:
        if condition not in self.CONDITIONS:
            raise ValueError(f"unknown condition {condition}")
        self.condition = condition
        self.validator = validator or VisibleContextSemanticValidator()
        self.correction_generator = correction_generator

    def _visible_prerequisite_action(
        self,
        observation: dict[str, Any],
        proposal: dict[str, Any],
        previous_tool_result: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Return a uniquely identifiable producer call for a missing runtime value.

        Example: a proposed reminder mutation contains an unsupported ``token``.
        If exactly one visible tool declares ``token`` as output and all of that
        tool's required inputs are uniquely supported by the user goal/receipt,
        call the producer first. No reference action or future result is used.
        """

        catalog = {str(entry.get("name", "")): entry for entry in observation.get("available_tools", [])}
        # With additional visible tools, the next-intent order is ambiguous.
        # A local data dependency does not prove that its producer is the next
        # globally correct action (e.g. open account -> stock -> login -> balance).
        if len(catalog) != 2:
            return None
        current_schema = catalog.get(str(proposal.get("tool", "")))
        if current_schema is None:
            return None
        arguments = dict(proposal.get("arguments", {}))
        unsupported: list[str] = []
        for parameter in current_schema.get("required", []):
            value = arguments.get(str(parameter))
            visible_values = self.validator.explicit_values(
                observation, str(parameter), previous_tool_result
            )
            if value is None or not any(_normalize(value) == _normalize(item) for item in visible_values):
                unsupported.append(str(parameter))

        candidates: list[dict[str, Any]] = []
        for missing_value in unsupported:
            producers = [
                schema
                for name, schema in catalog.items()
                if name != proposal.get("tool")
                and missing_value in dict(schema.get("output_parameters", {}))
            ]
            if len(producers) != 1:
                continue
            producer = producers[0]
            producer_arguments: dict[str, Any] = {}
            for parameter in producer.get("required", []):
                visible = self.validator.prerequisite_input_value(
                    observation, str(parameter), previous_tool_result
                )
                if visible is None:
                    producer_arguments = {}
                    break
                producer_arguments[str(parameter)] = visible
            if len(producer_arguments) != len(producer.get("required", [])):
                continue
            candidate = {"tool": producer["name"], "arguments": producer_arguments}
            if self.validator.validate(observation, candidate, previous_tool_result).semantic_valid == "true":
                candidates.append(candidate)
        unique = {
            json.dumps(canonical_action(candidate), ensure_ascii=False, sort_keys=True): candidate
            for candidate in candidates
        }
        return next(iter(unique.values())) if len(unique) == 1 else None

    def inspect(
        self,
        observation: dict[str, Any],
        proposal: dict[str, Any],
        previous_tool_result: dict[str, Any] | None = None,
    ) -> HarnessDecision:
        observation = public_harness_observation(observation)
        a_validity = self.validator.validate(observation, proposal, previous_tool_result)
        if self.condition == "H0" or a_validity.semantic_valid == "true":
            return HarnessDecision(False, "feedback", "none", None, None, True, False, False, False)

        if proposal.get("type") == "finish":
            return self._feedback_only("premature_finish", a_validity.reason)

        catalog = {str(entry.get("name", "")): entry for entry in observation.get("available_tools", [])}
        tool_name = str(proposal.get("tool", ""))
        schema = catalog.get(tool_name)
        if schema is None:
            return self._feedback_only("unknown_tool", a_validity.reason)

        prerequisite = self._visible_prerequisite_action(
            observation, proposal, previous_tool_result
        )
        if prerequisite is not None:
            reason = "unique visible producer supplies a required runtime value"
            if self.condition in {"H1", "Hplacebo"}:
                return HarnessDecision(
                    True, "feedback", "visible_prerequisite_repair", None, reason, True, False, False, False
                )
            return HarnessDecision(
                True,
                "replace",
                "visible_prerequisite_repair",
                canonical_action(prerequisite),
                reason,
                True,
                False,
                self.condition == "H3",
                False,
            )

        corrected = canonical_action(proposal)
        arguments = dict(corrected.get("arguments", {}))
        changed = False
        for parameter in schema.get("required", []):
            visible = self.validator.explicit_value(observation, str(parameter), previous_tool_result)
            receipt_has_value = bool(
                previous_tool_result
                and any(str(key).lower() == str(parameter).lower() for key in previous_tool_result)
            )
            if (
                parameter not in arguments
                and visible is not None
                and (
                    receipt_has_value
                    or self.validator.locally_supported_for_action(observation, visible, arguments)
                )
            ):
                arguments[str(parameter)] = visible
                changed = True
            elif (
                parameter in arguments
                and str(parameter).lower().endswith("_id")
                and visible is not None
                and _normalize(arguments[parameter]) != _normalize(visible)
            ):
                arguments[str(parameter)] = visible
                changed = True
            # Do not rewrite an existing argument from goal text alone.  A
            # multi-intent goal can mention several values with the same field
            # name; without a trusted receipt the association is ambiguous.
            # Existing-value repairs are therefore feedback-only/unknown in
            # this narrow first deployable harness.
        corrected["arguments"] = arguments

        suggested = (previous_tool_result or {}).get("suggested_action")
        if isinstance(suggested, dict):
            suggested_validity = self.validator.validate(observation, suggested, previous_tool_result)
            if suggested_validity.semantic_valid == "true":
                corrected = canonical_action(suggested)
                changed = corrected != canonical_action(proposal)

        generated_repair = False
        if (
            not changed
            and self.correction_generator is not None
            and schema is not None
            and a_validity.semantic_valid != "true"
        ):
            candidate = self.correction_generator(observation, proposal, a_validity.reason, previous_tool_result)
            if isinstance(candidate, dict) and candidate.get("tool") == proposal.get("tool"):
                candidate = canonical_action(candidate)
                candidate_validity = self.validator.validate(observation, candidate, previous_tool_result)
                if candidate_validity.semantic_valid == "true" and candidate != canonical_action(proposal):
                    corrected = candidate
                    changed = True
                    generated_repair = True

        b_validity = self.validator.validate(observation, corrected, previous_tool_result)
        if not changed or b_validity.semantic_valid != "true":
            return self._feedback_only("unresolved_visible_constraint", b_validity.reason)

        missing_before = "missing required arguments" in a_validity.reason
        patch_id = (
            "generated_visible_schema_repair"
            if generated_repair and missing_before
            else "generated_visible_argument_repair"
            if generated_repair
            else "visible_schema_repair"
            if missing_before
            else "visible_argument_repair"
        )
        event_type = "repair" if missing_before else "replace"
        if self.condition in {"H1", "Hplacebo"}:
            return HarnessDecision(True, "feedback", patch_id, None, b_validity.reason, True, False, False, False)
        return HarnessDecision(
            True,
            event_type,
            patch_id,
            corrected,
            b_validity.reason,
            True,
            False,
            self.condition == "H3",
            False,
        )

    def execute(
        self,
        observation: dict[str, Any],
        proposal: dict[str, Any],
        previous_tool_result: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], HarnessDecision]:
        decision = self.inspect(observation, proposal, previous_tool_result)
        if self.condition == "H3" and decision.corrected_action is not None and decision.changes_execution:
            return copy.deepcopy(decision.corrected_action), decision
        return proposal, decision

    def validity_pair(
        self,
        observation: dict[str, Any],
        proposal: dict[str, Any],
        correction: dict[str, Any],
        previous_tool_result: dict[str, Any] | None = None,
    ) -> tuple[ActionValidity, ActionValidity]:
        return (
            self.validator.validate(observation, proposal, previous_tool_result),
            self.validator.validate(observation, correction, previous_tool_result),
        )

    def _feedback_only(self, patch_id: str, reason: str) -> HarnessDecision:
        if self.condition == "H0":
            return HarnessDecision(False, "feedback", "none", None, None, True, False, False, False)
        return HarnessDecision(True, "feedback", patch_id, None, reason, True, False, False, False)


def audit_record(
    observation: dict[str, Any],
    proposal: dict[str, Any],
    correction: dict[str, Any] | None,
    a_validity: ActionValidity,
    b_validity: ActionValidity | None,
) -> dict[str, Any]:
    """Serializable evidence record that contains no hidden reference action."""

    return {
        "observation": public_harness_observation(observation),
        "proposal": canonical_action(proposal),
        "correction": canonical_action(correction) if correction else None,
        "a_executable_valid": a_validity.executable_valid,
        "a_semantic_valid": a_validity.semantic_valid,
        "b_executable_valid": b_validity.executable_valid if b_validity else None,
        "b_semantic_valid": b_validity.semantic_valid if b_validity else None,
        "validator_source": "visible_context_rules_v1",
        "record_digest": json.dumps(
            [public_harness_observation(observation), proposal, correction], ensure_ascii=False, sort_keys=True
        ),
    }
