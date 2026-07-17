from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence


_VISIBLE_IDENTIFIER = re.compile(r"(?<!\d)\d{6,}(?!\d)")


def visible_structure_reasons(task: dict[str, Any]) -> tuple[str, ...]:
    """Return reference-free structural reasons that a task may yield a repair.

    This function deliberately reads only the user-visible goal and tool schema.
    Reference actions, reference receipts, success predicates, and hidden labels
    must never affect curriculum membership.
    """

    goal = str(task.get("user_goal", ""))
    reasons: set[str] = set()

    visible_identifiers = tuple(dict.fromkeys(_VISIBLE_IDENTIFIER.findall(goal)))
    if len(visible_identifiers) == 1:
        reasons.add("unique_visible_identifier")

    # The original controlled-v1 data derived available_tools from the gold
    # action sequence. Schema structure is therefore usable only when the data
    # builder explicitly certifies reference-independent provenance.
    if task.get("available_tools_reference_independent") is not True:
        return tuple(sorted(reasons))

    raw_tools = task.get("available_tools", [])
    tools = [tool for tool in raw_tools if isinstance(tool, dict)]

    for producer in tools:
        outputs = {
            str(name)
            for name in dict(producer.get("output_parameters", {}))
            if isinstance(name, str)
        }
        if not outputs:
            continue
        producer_name = str(producer.get("name", ""))
        for consumer in tools:
            if str(consumer.get("name", "")) == producer_name:
                continue
            required = {
                str(name)
                for name in consumer.get("required", [])
                if isinstance(name, str)
            }
            if outputs & required:
                reasons.add("visible_tool_dependency")
                break
        if "visible_tool_dependency" in reasons:
            break

    return tuple(sorted(reasons))


def _stable_order(tasks: Sequence[dict[str, Any]], seed: int, namespace: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        sorted(
            tasks,
            key=lambda task: hashlib.sha256(
                f"{namespace}:{seed}:{task.get('task_id', '')}".encode("utf-8")
            ).hexdigest(),
        )
    )


def _pool_hash(tasks: Sequence[dict[str, Any]]) -> str:
    payload = "\n".join(
        sorted(
            f"{task.get('task_id', '')}:{','.join(visible_structure_reasons(task))}"
            for task in tasks
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CurriculumSelection:
    task: dict[str, Any]
    source: str
    reasons: tuple[str, ...]
    stratum: str | None = None


class VisibleStructureCurriculum:
    """Deterministic mixture of the original split and a visible-only pool."""

    def __init__(self, tasks: Sequence[dict[str, Any]], fraction: float, seed: int):
        if not tasks:
            raise ValueError("curriculum requires at least one task")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("visible curriculum fraction must be in [0, 1]")

        task_ids = [str(task.get("task_id", "")) for task in tasks]
        if any(not task_id for task_id in task_ids):
            raise ValueError("curriculum tasks require non-empty task_id values")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("curriculum task_id values must be unique")

        self.fraction = float(fraction)
        self.seed = int(seed)
        self._original = tuple(tasks)
        self._base = _stable_order(tasks, seed, "base")
        self._visible = _stable_order(
            [task for task in tasks if visible_structure_reasons(task)],
            seed,
            "visible",
        )
        self._visible_by_reason = {
            reason: _stable_order(
                [task for task in tasks if reason in visible_structure_reasons(task)],
                seed,
                f"visible:{reason}",
            )
            for reason in ("unique_visible_identifier", "visible_tool_dependency")
        }
        self._visible_by_reason = {
            reason: pool for reason, pool in self._visible_by_reason.items() if pool
        }
        self._strata = tuple(sorted(self._visible_by_reason))
        if self.fraction > 0.0 and not self._visible:
            raise ValueError("visible curriculum requested but no visible-structure tasks were found")
        self._ratio = Fraction(str(self.fraction)).limit_denominator(1000)

    @property
    def visible_pool_size(self) -> int:
        return len(self._visible)

    @property
    def visible_pool_hash(self) -> str:
        return _pool_hash(self._visible)

    @property
    def reference_free_selection(self) -> bool:
        return True

    @property
    def reason_pool_sizes(self) -> dict[str, int]:
        return {reason: len(pool) for reason, pool in self._visible_by_reason.items()}

    def _visible_selection(self, arm_index: int, task_offset: int = 0) -> CurriculumSelection:
        stratum_index, within_round = divmod(arm_index, len(self._strata))
        reason = self._strata[within_round]
        pool = self._visible_by_reason[reason]
        task = pool[(stratum_index + task_offset) % len(pool)]
        return CurriculumSelection(
            task,
            "visible_curriculum",
            visible_structure_reasons(task),
            stratum=reason,
        )

    def _select_with_offset(self, global_slot: int, task_offset: int) -> CurriculumSelection:
        if global_slot < 0:
            raise ValueError("global_slot must be non-negative")
        if self._ratio.numerator == 0:
            task = self._original[(global_slot + task_offset) % len(self._original)]
            return CurriculumSelection(task, "sequential", visible_structure_reasons(task))
        if self._ratio.numerator == self._ratio.denominator:
            return self._visible_selection(global_slot, task_offset)

        cycle, position = divmod(global_slot, self._ratio.denominator)
        if position < self._ratio.numerator:
            arm_index = cycle * self._ratio.numerator + position
            return self._visible_selection(arm_index, task_offset)

        base_width = self._ratio.denominator - self._ratio.numerator
        arm_index = cycle * base_width + position - self._ratio.numerator
        task = self._base[(arm_index + task_offset) % len(self._base)]
        return CurriculumSelection(task, "base_mix", visible_structure_reasons(task))

    def select(self, global_slot: int) -> CurriculumSelection:
        return self._select_with_offset(global_slot, 0)

    def select_batch(self, update: int, world_size: int) -> tuple[CurriculumSelection, ...]:
        """Select one deterministic, task-unique assignment per DDP rank."""

        if update < 0:
            raise ValueError("update must be non-negative")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if world_size > len(self._original):
            raise ValueError("world_size cannot exceed the number of training tasks")

        used_task_ids: set[str] = set()
        selections: list[CurriculumSelection] = []
        for rank in range(world_size):
            global_slot = update * world_size + rank
            for task_offset in range(len(self._original)):
                selection = self._select_with_offset(global_slot, task_offset)
                task_id = str(selection.task.get("task_id", ""))
                if not task_id:
                    raise ValueError("curriculum tasks require non-empty task_id values")
                if task_id not in used_task_ids:
                    used_task_ids.add(task_id)
                    selections.append(selection)
                    break
            else:
                raise RuntimeError("unable to construct a task-unique DDP curriculum batch")
        return tuple(selections)

    def assignment_sequence_hash(self, num_updates: int, world_size: int) -> str:
        if num_updates < 0:
            raise ValueError("num_updates must be non-negative")
        rows = []
        for update in range(num_updates):
            for rank, selection in enumerate(self.select_batch(update, world_size)):
                rows.append(
                    ":".join(
                        [
                            str(update),
                            str(rank),
                            str(selection.task["task_id"]),
                            selection.source,
                            selection.stratum or "none",
                        ]
                    )
                )
        return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
