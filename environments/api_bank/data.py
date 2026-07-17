from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def digest_records(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reference_years_consistent(task: dict[str, Any]) -> bool:
    """Reject trace labels whose explicit year contradicts the user goal."""

    goal_years = set(re.findall(r"\b(?:19|20)\d{2}\b", str(task.get("user_goal", ""))))
    action_years = {
        year
        for action in task.get("reference_actions", [])
        for value in dict(action.get("arguments", {})).values()
        for year in re.findall(r"\b(?:19|20)\d{2}\b", str(value))
    }
    return not goal_years or not action_years or action_years.issubset(goal_years)


def parse_api_catalog(api_dir: str | Path) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(api_dir).glob("*.py")):
        if path.name in {"api.py", "__init__.py"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            input_parameters: dict[str, Any] = {}
            output_parameters: dict[str, Any] = {}
            description = ""
            for item in node.body:
                if not isinstance(item, (ast.Assign, ast.AnnAssign)):
                    continue
                target = item.targets[0] if isinstance(item, ast.Assign) else item.target
                value = item.value
                if isinstance(target, ast.Name) and target.id in {
                    "input_parameters",
                    "output_parameters",
                    "description",
                }:
                    try:
                        literal = ast.literal_eval(value)
                    except (ValueError, TypeError):
                        continue
                    if target.id == "input_parameters" and isinstance(literal, dict):
                        input_parameters = literal
                    elif target.id == "output_parameters" and isinstance(literal, dict):
                        output_parameters = literal
                    elif target.id == "description" and isinstance(literal, str):
                        description = literal
            if input_parameters:
                catalog[node.name] = {
                    "name": node.name,
                    "description": description,
                    "required": sorted(input_parameters),
                    "optional": [],
                    "parameters": input_parameters,
                    "output_parameters": output_parameters,
                    "source_file": path.name,
                }
    return catalog


def candidate_tool_names_from_source(
    path: Path,
    catalog: dict[str, dict[str, Any]],
) -> tuple[list[str], str]:
    """Build the runtime tool set without consulting reference actions.

    API-Bank scenario filenames declare the primary tools. We add schema-only
    producers for runtime identifiers (tokens and ``*_id`` values), plus the
    level-3 ToolSearcher. This makes the public tool set invariant to changes
    in the gold trace while retaining the prerequisites needed for execution.
    """

    stem = path.stem
    scenario_prefix = re.split(r"-level-\d", stem, maxsplit=1)[0]
    selected: set[str] = set()
    for token in scenario_prefix.split("-"):
        if token in catalog:
            selected.add(token)
            continue
        query_alias = f"Query{token[3:]}" if token.startswith("Get") else ""
        if query_alias in catalog:
            selected.add(query_alias)

    if "-level-3-" in stem and "ToolSearcher" in catalog:
        selected.add("ToolSearcher")

    if not selected:
        # Reference-independent fallback for malformed or synthetic filenames.
        return sorted(catalog), "global_catalog_fallback"

    for _ in range(3):
        dependency_fields = {
            field
            for name in selected
            for field in catalog[name].get("required", [])
            if field == "token" or str(field).endswith("_id")
        }
        producers = {
            name
            for name, schema in catalog.items()
            if set(schema.get("output_parameters", {})) & dependency_fields
        }
        expanded = selected | producers
        if expanded == selected:
            break
        selected = expanded
    return sorted(selected), "source_filename_plus_schema_prerequisites"


def parse_dialogue(path: Path, catalog: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            return None
    calls = [record for record in records if record.get("role") == "API"]
    user_turns = [record.get("text", "") for record in records if record.get("role") == "User"]
    if not calls or not user_turns:
        return None
    reference_actions = [
        {"tool": call["api_name"], "arguments": call.get("param_dict", {})}
        for call in calls
        if call.get("api_name") in catalog and call.get("result", {}).get("exception") is None
    ]
    if len(reference_actions) != len(calls):
        return None
    reference_tool_receipts: list[dict[str, Any]] = []
    for call in calls:
        output = call.get("result", {}).get("output")
        receipt: dict[str, Any] = {"status": "ok", "tool": call["api_name"]}
        if isinstance(output, dict):
            receipt.update(output)
        elif output is not None:
            receipt["output"] = output
        reference_tool_receipts.append(receipt)
    reference_tools = sorted({action["tool"] for action in reference_actions})
    public_tools, public_tool_provenance = candidate_tool_names_from_source(path, catalog)
    relative = path.as_posix()
    source_id = stable_hash(relative)[:16]
    goal = "\n".join(str(turn) for turn in user_turns)
    signature = "|".join(action["tool"] + ":" + ",".join(sorted(action["arguments"])) for action in reference_actions)
    return {
        "source_sample_id": source_id,
        "source_path": relative,
        "api_family_id": reference_tools[0] if len(reference_tools) == 1 else "+".join(reference_tools),
        "user_goal": goal,
        "normalized_goal_template": re.sub(r"\d+|[^a-z]+", " ", goal.lower()).strip(),
        "reference_action_signature": signature,
        "available_tools": [catalog[name] for name in public_tools],
        "available_tools_provenance": public_tool_provenance,
        "available_tools_reference_independent": True,
        "initial_state": {},
        "reference_actions": reference_actions,
        # Private environment replay data. It is exposed only after the agent
        # executes the corresponding correct tool call, never in observation().
        "reference_tool_receipts": reference_tool_receipts,
        "success_predicate": {"type": "reference_action_sequence", "target": reference_actions},
        "eligible_patches": ["missing_required_argument", "wrong_tool_replace", "premature_finish"],
        "max_steps": max(12, len(reference_actions) + 2),
    }


def assign_splits(tasks: list[dict[str, Any]], seed: int = 20260714) -> dict[str, list[dict[str, Any]]]:
    tools = sorted({action["tool"] for task in tasks for action in task["reference_actions"]})
    ood_tools = {tool for tool in tools if int(stable_hash(f"{seed}:tool:{tool}")[:8], 16) / 0xFFFFFFFF < 0.20}
    id_pool: list[dict[str, Any]] = []
    ood_tasks: list[dict[str, Any]] = []
    split_conflicts: list[dict[str, Any]] = []
    for task in tasks:
        task_tools = {action["tool"] for action in task["reference_actions"]}
        if task_tools and task_tools <= ood_tools:
            ood_tasks.append(task)
        elif task_tools.isdisjoint(ood_tools):
            id_pool.append(task)
        else:
            split_conflicts.append(task)
    parents = list(range(len(id_pool)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    by_goal: dict[str, int] = {}
    by_actions: dict[str, int] = {}
    for index, task in enumerate(id_pool):
        goal = task["normalized_goal_template"]
        actions = json.dumps(task["reference_actions"], ensure_ascii=False, sort_keys=True)
        if goal in by_goal:
            union(index, by_goal[goal])
        else:
            by_goal[goal] = index
        if actions in by_actions:
            union(index, by_actions[actions])
        else:
            by_actions[actions] = index
    groups: dict[int, list[dict[str, Any]]] = {}
    for index, task in enumerate(id_pool):
        groups.setdefault(find(index), []).append(task)
    splits = {
        "train": [],
        "dev": [],
        "test_id": [],
        "test_tool_ood": ood_tasks,
        "split_conflict_excluded": split_conflicts,
    }
    for key in sorted(groups):
        group_signature = sorted(task["source_sample_id"] for task in groups[key])
        value = int(stable_hash(f"{seed}:{group_signature}")[:8], 16) / 0xFFFFFFFF
        split = "train" if value < 0.70 else "dev" if value < 0.80 else "test_id"
        splits[split].extend(groups[key])
    for split, records in splits.items():
        for index, task in enumerate(sorted(records, key=lambda item: item["source_sample_id"])):
            task["task_id"] = f"apibank_ctrl_{split}_{index:06d}"
    return splits
