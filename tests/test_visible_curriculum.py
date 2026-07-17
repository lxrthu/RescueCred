from __future__ import annotations

import copy

import pytest

from rescuecredit.visible_curriculum import VisibleStructureCurriculum, visible_structure_reasons


class ReferenceTripwire(dict):
    forbidden = {"reference_actions", "reference_tool_receipts", "success_predicate"}

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"curriculum read forbidden field: {key}")
        return super().get(key, default)

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"curriculum read forbidden field: {key}")
        return super().__getitem__(key)


def dependency_task(task_id: str = "dependency") -> dict:
    return {
        "task_id": task_id,
        "user_goal": "Please authenticate me and then read my balance.",
        "available_tools": [
            {
                "name": "Login",
                "required": ["username", "password"],
                "output_parameters": {"token": {"type": "str"}},
            },
            {
                "name": "QueryBalance",
                "required": ["token"],
                "output_parameters": {"balance": {"type": "float"}},
            },
        ],
        "available_tools_reference_independent": True,
        "reference_actions": [{"tool": "DO_NOT_READ", "arguments": {}}],
        "reference_tool_receipts": [{"secret": "DO_NOT_READ"}],
        "success_predicate": {"hidden": "DO_NOT_READ"},
    }


def identifier_task(task_id: str = "identifier") -> dict:
    return {
        "task_id": task_id,
        "user_goal": "Cancel appointment 1234567890.",
        "available_tools": [
            {
                "name": "CancelRegistration",
                "required": ["appointment_id"],
                "output_parameters": {"status": {"type": "str"}},
            }
        ],
    }


def plain_task(task_id: str = "plain") -> dict:
    return {
        "task_id": task_id,
        "user_goal": "Tell me the weather.",
        "available_tools": [
            {
                "name": "Weather",
                "required": ["city"],
                "output_parameters": {"temperature": {"type": "float"}},
            }
        ],
    }


def test_visible_structure_detection_uses_goal_and_schema():
    assert visible_structure_reasons(identifier_task()) == ("unique_visible_identifier",)
    assert visible_structure_reasons(dependency_task()) == ("visible_tool_dependency",)
    assert visible_structure_reasons(plain_task()) == ()


def test_schema_reason_requires_reference_independent_tool_provenance():
    task = dependency_task()
    task["available_tools_reference_independent"] = False
    assert visible_structure_reasons(task) == ()


def test_reference_fields_cannot_change_membership_or_pool_hash():
    original = dependency_task()
    mutated = copy.deepcopy(original)
    mutated["reference_actions"] = [{"tool": "Anything", "arguments": {"x": 1}}]
    mutated["reference_tool_receipts"] = [{"future": "different"}]
    mutated["success_predicate"] = {"target": "different"}

    assert visible_structure_reasons(original) == visible_structure_reasons(mutated)
    assert VisibleStructureCurriculum([original], 1.0, 42).visible_pool_hash == (
        VisibleStructureCurriculum([mutated], 1.0, 42).visible_pool_hash
    )


def test_reference_fields_are_never_read():
    guarded = ReferenceTripwire(dependency_task())
    assert visible_structure_reasons(guarded) == ("visible_tool_dependency",)
    sampler = VisibleStructureCurriculum([guarded], 1.0, 42)
    assert sampler.select(0).task["task_id"] == "dependency"


def test_half_mix_is_exact_and_deterministic():
    tasks = [dependency_task(), identifier_task(), plain_task()]
    first = VisibleStructureCurriculum(tasks, 0.5, 42)
    second = VisibleStructureCurriculum(tasks, 0.5, 42)

    selections = [first.select(slot) for slot in range(12)]
    assert sum(item.source == "visible_curriculum" for item in selections) == 6
    assert [item.stratum for item in selections if item.source == "visible_curriculum"] == [
        "unique_visible_identifier",
        "visible_tool_dependency",
        "unique_visible_identifier",
        "visible_tool_dependency",
        "unique_visible_identifier",
        "visible_tool_dependency",
    ]
    assert [item.task["task_id"] for item in selections] == [
        second.select(slot).task["task_id"] for slot in range(12)
    ]
    assert all(item.reasons for item in selections if item.source == "visible_curriculum")


def test_zero_fraction_preserves_legacy_sequential_order():
    tasks = [plain_task("a"), dependency_task("b"), identifier_task("c")]
    sampler = VisibleStructureCurriculum(tasks, 0.0, 7)
    assert [sampler.select(slot).task["task_id"] for slot in range(5)] == ["a", "b", "c", "a", "b"]
    assert all(sampler.select(slot).source == "sequential" for slot in range(5))


def test_ddp_batches_never_repeat_a_task_within_an_update():
    tasks = [
        dependency_task(f"dependency-{index}") for index in range(8)
    ] + [
        identifier_task(f"identifier-{index}") for index in range(8)
    ] + [
        plain_task(f"plain-{index}") for index in range(8)
    ]
    sampler = VisibleStructureCurriculum(tasks, 0.75, 42)
    for update in range(100):
        batch = sampler.select_batch(update, world_size=4)
        task_ids = [item.task["task_id"] for item in batch]
        assert len(task_ids) == len(set(task_ids))


def test_ddp_batch_preserves_requested_source_ratio():
    tasks = [dependency_task(f"dependency-{index}") for index in range(8)] + [
        identifier_task(f"identifier-{index}") for index in range(8)
    ]
    sampler = VisibleStructureCurriculum(tasks, 0.75, 42)
    selections = [item for update in range(8) for item in sampler.select_batch(update, 4)]
    assert sum(item.source == "visible_curriculum" for item in selections) == 24
    assert sampler.assignment_sequence_hash(8, 4) == sampler.assignment_sequence_hash(8, 4)
    assert sampler.assignment_sequence_hash(8, 4) != VisibleStructureCurriculum(
        tasks, 0.5, 42
    ).assignment_sequence_hash(8, 4)


def test_fraction_requires_visible_pool_and_valid_range():
    with pytest.raises(ValueError, match="no visible-structure tasks"):
        VisibleStructureCurriculum([plain_task()], 0.5, 42)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        VisibleStructureCurriculum([dependency_task()], 1.1, 42)
    with pytest.raises(ValueError, match="must be unique"):
        VisibleStructureCurriculum([dependency_task("same"), plain_task("same")], 0.5, 42)
