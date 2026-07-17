from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable


CandidateSelector = Callable[[dict[str, Any]], int | None]


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _nested_values(value: Any, parameter: str) -> tuple[list[Any], list[Any]]:
    exact: list[Any] = []
    all_values: list[Any] = []
    target = _normalize_key(parameter)
    if isinstance(value, dict):
        for key, child in value.items():
            if _scalar(child):
                all_values.append(child)
                if _normalize_key(str(key)) == target:
                    exact.append(child)
            child_exact, child_all = _nested_values(child, parameter)
            exact.extend(child_exact)
            all_values.extend(child_all)
    elif isinstance(value, (list, tuple)):
        for child in value:
            child_exact, child_all = _nested_values(child, parameter)
            exact.extend(child_exact)
            all_values.extend(child_all)
    elif _scalar(value):
        all_values.append(value)
    return exact, all_values


def _key_tokens(value: str) -> set[str]:
    generic = {"id", "ids", "value", "values", "data", "result", "status"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
        if token not in generic
    }


def _receipt_evidence(
    value: Any,
    parameter: str,
    path: tuple[str, ...] = (),
) -> list[tuple[Any, str, str]]:
    evidence: list[tuple[Any, str, str]] = []
    parameter_key = _normalize_key(parameter)
    parameter_tokens = _key_tokens(parameter)
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = (*path, key_text)
            if _scalar(child):
                if _normalize_key(key_text) == parameter_key:
                    source = "exact_receipt_key"
                elif parameter_tokens and parameter_tokens & _key_tokens(key_text):
                    source = "related_receipt_key"
                else:
                    source = "receipt_any"
                origin = "receipt." + ".".join(child_path)
                evidence.append((child, source, origin))
            evidence.extend(_receipt_evidence(child, parameter, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            evidence.extend(_receipt_evidence(child, parameter, (*path, str(index))))
    return evidence


def _instruction_values(instruction: str) -> list[str]:
    patterns = [
        r'"([^"\n]{1,200})"',
        r"'([^'\n]{1,200})'",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"\bhttps?://[^\s,;]+",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9])",
    ]
    values: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, instruction):
            value = match.group(1) if match.lastindex else match.group(0)
            value = value.strip().rstrip(".")
            if value:
                values.append(value)
    return values


def _instruction_evidence(
    instruction: str, parameter: str
) -> list[tuple[str, str, str]]:
    values = _instruction_values(instruction)
    label = parameter.lower().replace("_", " ")
    label_positions = [match.start() for match in re.finditer(re.escape(label), instruction.lower())]
    evidence: list[tuple[str, str, str]] = []
    for value in values:
        positions = [match.start() for match in re.finditer(re.escape(value), instruction)]
        strong = bool(
            label_positions
            and positions
            and min(abs(left - right) for left in label_positions for right in positions) <= 120
        )
        position = positions[0] if positions else 0
        start = max(0, position - 80)
        end = min(len(instruction), position + len(value) + 80)
        context = re.sub(r"\s+", " ", instruction[start:end]).strip()
        evidence.append(
            (
                value,
                "instruction_labeled" if strong else "instruction_any",
                f"instruction_span[{start}:{end}]={context}",
            )
        )
    return evidence


def _deduplicate(values: list[Any], limit: int = 64) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if not _scalar(value):
            continue
        marker = f"{type(value).__name__}:{value!s}"
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(value)
        if len(unique) >= limit:
            break
    return unique


@dataclass(frozen=True)
class CandidateRepairDecision:
    changed: bool
    patch_id: str
    candidate_count: int
    selected_by: str | None
    selected_sources: tuple[str, ...] = ()
    selected_origins: tuple[str, ...] = ()


class AppWorldCandidateHarness:
    """Reference-free missing-argument repair by visible candidate selection."""

    def __init__(
        self,
        selector: CandidateSelector | None = None,
        min_selector_candidates: int = 1,
    ) -> None:
        self.selector = selector
        self.min_selector_candidates = max(1, int(min_selector_candidates))

    def candidates(
        self,
        instruction: str,
        receipts: dict[str, Any] | None,
        parameter: str,
    ) -> tuple[list[Any], list[Any]]:
        details = self.candidate_details(instruction, receipts, parameter)
        exact = [
            item["value"]
            for item in details
            if "exact_receipt_key" in item["sources"]
        ]
        candidates = [item["value"] for item in details]
        return exact, candidates

    def candidate_details(
        self,
        instruction: str,
        receipts: dict[str, Any] | None,
        parameter: str,
    ) -> list[dict[str, Any]]:
        raw = _receipt_evidence(receipts or {}, parameter)
        raw.extend(_instruction_evidence(instruction, parameter))
        details: list[dict[str, Any]] = []
        marker_to_index: dict[str, int] = {}
        for value, source, origin in raw:
            if not _scalar(value):
                continue
            marker = f"{type(value).__name__}:{value!s}"
            if marker in marker_to_index:
                sources = details[marker_to_index[marker]]["sources"]
                if source not in sources:
                    sources.append(source)
                origins = details[marker_to_index[marker]]["origins"]
                if origin not in origins:
                    origins.append(origin)
                continue
            marker_to_index[marker] = len(details)
            details.append(
                {"value": value, "sources": [source], "origins": [origin]}
            )
            if len(details) >= 64:
                break
        return details

    def repair(
        self,
        instruction: str,
        receipts: dict[str, Any] | None,
        proposal: dict[str, Any],
        required_fields: list[str],
        public_schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], CandidateRepairDecision]:
        arguments = dict(proposal.get("arguments", {}))
        missing = sorted(set(required_fields) - set(arguments))
        if len(missing) != 1:
            return copy.deepcopy(proposal), CandidateRepairDecision(
                False, "none" if not missing else "ambiguous_missing_fields", 0, None
            )
        parameter = missing[0]
        details = self.candidate_details(instruction, receipts, parameter)
        candidates = [item["value"] for item in details]
        exact = [
            item["value"]
            for item in details
            if "exact_receipt_key" in item["sources"]
        ]
        selected_index: int | None = None
        selected_by: str | None = None
        if len(exact) == 1:
            selected_index = candidates.index(exact[0])
            selected_by = "unique_receipt_key"
        elif (
            self.selector is not None
            and len(candidates) >= self.min_selector_candidates
        ):
            selected_index = self.selector(
                {
                    "instruction": instruction,
                    # Candidate values are extracted from receipts locally;
                    # the selector only needs the bounded candidate list.
                    "receipts": None,
                    "tool": proposal.get("tool"),
                    "parameter": parameter,
                    "candidates": candidates,
                    "candidate_sources": [item["sources"] for item in details],
                    "candidate_origins": [item["origins"] for item in details],
                    "public_schema": public_schema or {},
                }
            )
            selected_by = "frozen_model_candidate_index"
        elif self.selector is not None and candidates:
            return copy.deepcopy(proposal), CandidateRepairDecision(
                False,
                "insufficient_visible_context",
                len(candidates),
                None,
            )
        if selected_index is None or not 0 <= selected_index < len(candidates):
            return copy.deepcopy(proposal), CandidateRepairDecision(
                False, "no_supported_candidate", len(candidates), None
            )
        strong_sources = {
            "exact_receipt_key",
            "related_receipt_key",
            "instruction_labeled",
        }
        if not strong_sources.intersection(details[selected_index]["sources"]):
            return copy.deepcopy(proposal), CandidateRepairDecision(
                False, "weak_candidate_evidence", len(candidates), None
            )
        corrected = copy.deepcopy(proposal)
        corrected.setdefault("arguments", {})[parameter] = copy.deepcopy(
            candidates[selected_index]
        )
        return corrected, CandidateRepairDecision(
            True,
            "visible_candidate_repair",
            len(candidates),
            selected_by,
            tuple(details[selected_index]["sources"]),
            tuple(details[selected_index]["origins"]),
        )
