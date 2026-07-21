from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


LABELS = ("rescue_preference", "reverse_preference")
PUBLIC_FIELDS = (
    "event_id",
    "task_id_hash",
    "source",
    "scenario_name",
    "reference_free_prefix_steps",
    "action_a",
    "action_b",
    "treatment_visible_history",
    "treatment_public_tool_schemas",
)
PRIVATE_FIELDS = (
    "event_id",
    "task_id_hash",
    "source",
    "decision",
    "decision_basis",
    "decision_value",
    "causal_weight",
)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tool(action: Mapping[str, Any]) -> str:
    return str(action.get("tool", action.get("recipient", "UNKNOWN")))


def _value_shape(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return ["list", sorted({stable_json(_value_shape(item)) for item in value})]
    if isinstance(value, Mapping):
        return {
            str(key): _value_shape(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return type(value).__name__


def _action_shape(action: Mapping[str, Any]) -> dict[str, Any]:
    arguments = action.get("arguments", {})
    return {
        "tool": _tool(action),
        "arguments": _value_shape(arguments if isinstance(arguments, Mapping) else {}),
    }


def _prefix_tools(history: Any) -> list[str]:
    if not isinstance(history, list):
        return []
    result: list[str] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        for key in ("tool", "tool_name", "recipient"):
            if item.get(key):
                result.append(str(item[key]))
                break
    return result


def _scenario_family(name: str) -> str:
    family = re.split(
        r"_(?:cellular|wifi|location|low_battery|distraction_tools)(?:_|$)",
        name,
        maxsplit=1,
    )[0]
    return re.sub(r"_\d+$", "", family)


def _schema_shapes(schemas: Any, tools: set[str]) -> list[Any]:
    if not isinstance(schemas, list):
        return []
    selected = []
    for schema in schemas:
        if not isinstance(schema, Mapping):
            continue
        name = str(schema.get("name", schema.get("tool", schema.get("recipient", ""))))
        if name and name not in tools:
            continue
        selected.append(_value_shape(schema))
    return sorted(selected, key=stable_json)


def public_projection(row: Mapping[str, Any], source: str) -> dict[str, Any]:
    missing = {
        "event_id",
        "task_id_hash",
        "action_a",
        "action_b",
        "treatment_visible_history",
        "treatment_public_tool_schemas",
    } - set(row)
    if missing:
        raise ValueError(f"candidate event lacks public fields: {sorted(missing)}")
    action_a = row["action_a"]
    action_b = row["action_b"]
    if not isinstance(action_a, Mapping) or not isinstance(action_b, Mapping):
        raise ValueError("candidate actions must be mappings")
    return {
        "event_id": str(row["event_id"]),
        "task_id_hash": str(row["task_id_hash"]),
        "source": source,
        "scenario_name": str(row.get("scenario_name", "")),
        "reference_free_prefix_steps": int(row.get("reference_free_prefix_steps", 0)),
        "action_a": dict(action_a),
        "action_b": dict(action_b),
        "treatment_visible_history": row["treatment_visible_history"],
        "treatment_public_tool_schemas": row["treatment_public_tool_schemas"],
    }


def private_projection(row: Mapping[str, Any], source: str) -> dict[str, Any]:
    decision = str(row.get("decision", ""))
    if decision not in LABELS:
        raise ValueError(f"unsupported compensation label: {decision}")
    return {
        "event_id": str(row["event_id"]),
        "task_id_hash": str(row["task_id_hash"]),
        "source": source,
        "decision": decision,
        "decision_basis": str(row.get("decision_basis", "")),
        "decision_value": float(row.get("decision_value", 0.0)),
        "causal_weight": float(row.get("causal_weight", 1.0)),
    }


def exact_signature_payload(public: Mapping[str, Any]) -> dict[str, Any]:
    action_a = public["action_a"]
    action_b = public["action_b"]
    if not isinstance(action_a, Mapping) or not isinstance(action_b, Mapping):
        raise ValueError("public actions must be mappings")
    return {
        "prefix_tools": _prefix_tools(public.get("treatment_visible_history")),
        "action_a": _action_shape(action_a),
        "action_b": _action_shape(action_b),
        "tool_relation": "same" if _tool(action_a) == _tool(action_b) else "different",
        "relevant_schema_shapes": _schema_shapes(
            public.get("treatment_public_tool_schemas"),
            {_tool(action_a), _tool(action_b)},
        ),
    }


def _tokens(public: Mapping[str, Any]) -> Counter[str]:
    text = stable_json(
        {
            key: public[key]
            for key in PUBLIC_FIELDS
            if key
            in {
                "action_a",
                "action_b",
                "treatment_visible_history",
                "treatment_public_tool_schemas",
            }
        }
    ).lower()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    compact = re.sub(r"\s+", " ", text)
    grams = [f"g4:{compact[index:index + 4]}" for index in range(max(0, len(compact) - 3))]
    return Counter([f"w:{word}" for word in words] + grams)


def _tfidf_vectors(public_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    counts = [_tokens(row) for row in public_rows]
    document_frequency: Counter[str] = Counter()
    for row in counts:
        document_frequency.update(row.keys())
    n = len(counts)
    vectors: list[dict[str, float]] = []
    for row in counts:
        weighted = {
            token: (1.0 + math.log(count))
            * (math.log((n + 1.0) / (document_frequency[token] + 1.0)) + 1.0)
            for token, count in row.items()
        }
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        vectors.append({token: value / norm for token, value in weighted.items()})
    return vectors


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def build_collision_audit(
    public_rows: Sequence[Mapping[str, Any]],
    private_rows: Sequence[Mapping[str, Any]],
    *,
    similarity_threshold: float = 0.90,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity threshold must be in (0,1]")
    public_by_id = {str(row["event_id"]): row for row in public_rows}
    private_by_id = {str(row["event_id"]): row for row in private_rows}
    if len(public_by_id) != len(public_rows) or len(private_by_id) != len(private_rows):
        raise ValueError("duplicate event IDs in compensation benchmark")
    if set(public_by_id) != set(private_by_id):
        raise ValueError("public/private event sets differ")
    ordered_ids = sorted(public_by_id)
    exact_groups: dict[str, list[str]] = defaultdict(list)
    exact_payloads: dict[str, dict[str, Any]] = {}
    for event_id in ordered_ids:
        payload = exact_signature_payload(public_by_id[event_id])
        signature = sha256_text(stable_json(payload))
        exact_groups[signature].append(event_id)
        exact_payloads[signature] = payload
    exact_pairs: list[dict[str, Any]] = []
    exact_mixed_events = 0
    conditional_errors = 0
    for signature, event_ids in sorted(exact_groups.items()):
        labels = Counter(private_by_id[event_id]["decision"] for event_id in event_ids)
        conditional_errors += min(labels.get(LABELS[0], 0), labels.get(LABELS[1], 0))
        if not all(labels.get(label, 0) for label in LABELS):
            continue
        exact_mixed_events += len(event_ids)
        rescue_ids = [event_id for event_id in event_ids if private_by_id[event_id]["decision"] == LABELS[0]]
        reverse_ids = [event_id for event_id in event_ids if private_by_id[event_id]["decision"] == LABELS[1]]
        for left in rescue_ids:
            for right in reverse_ids:
                if public_by_id[left]["task_id_hash"] == public_by_id[right]["task_id_hash"]:
                    continue
                exact_pairs.append(
                    {
                        "kind": "exact_signature",
                        "left_event_id": left,
                        "right_event_id": right,
                        "left_task_id_hash": str(public_by_id[left]["task_id_hash"]),
                        "right_task_id_hash": str(public_by_id[right]["task_id_hash"]),
                        "left_label": LABELS[0],
                        "right_label": LABELS[1],
                        "similarity": 1.0,
                        "signature_sha256": signature,
                        "signature": exact_payloads[signature],
                    }
                )
    ordered_public = [public_by_id[event_id] for event_id in ordered_ids]
    vectors = _tfidf_vectors(ordered_public)
    labels = [str(private_by_id[event_id]["decision"]) for event_id in ordered_ids]
    tasks = [str(public_by_id[event_id]["task_id_hash"]) for event_id in ordered_ids]
    nearest: dict[int, tuple[int, float]] = {}
    for left in range(len(ordered_ids)):
        candidates = []
        for right in range(len(ordered_ids)):
            if left == right or tasks[left] == tasks[right]:
                continue
            candidates.append((_cosine(vectors[left], vectors[right]), ordered_ids[right], right))
        if candidates:
            similarity, _, right = max(candidates, key=lambda item: (item[0], item[1]))
            nearest[left] = (right, similarity)
    approximate_pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for left, (right, similarity) in nearest.items():
        if (
            similarity < similarity_threshold
            or nearest.get(right, (-1, 0.0))[0] != left
            or labels[left] == labels[right]
        ):
            continue
        pair = tuple(sorted((left, right)))
        if pair in seen:
            continue
        seen.add(pair)
        approximate_pairs.append(
            {
                "kind": "cross_task_mutual_nearest",
                "left_event_id": ordered_ids[pair[0]],
                "right_event_id": ordered_ids[pair[1]],
                "left_task_id_hash": tasks[pair[0]],
                "right_task_id_hash": tasks[pair[1]],
                "left_label": labels[pair[0]],
                "right_label": labels[pair[1]],
                "similarity": similarity,
            }
        )
    pair_rows = sorted(
        exact_pairs + approximate_pairs,
        key=lambda row: (str(row["kind"]), str(row["left_event_id"]), str(row["right_event_id"])),
    )
    exact_tasks = {
        str(row[key])
        for row in exact_pairs
        for key in ("left_task_id_hash", "right_task_id_hash")
    }
    approximate_tasks = {
        str(row[key])
        for row in approximate_pairs
        for key in ("left_task_id_hash", "right_task_id_hash")
    }
    summary = {
        "events": len(ordered_ids),
        "tasks": len(set(tasks)),
        "label_counts": dict(sorted(Counter(labels).items())),
        "exact_signature_classes": len(exact_groups),
        "exact_mixed_classes": sum(
            len({private_by_id[event_id]["decision"] for event_id in event_ids}) == 2
            for event_ids in exact_groups.values()
        ),
        "exact_mixed_events": exact_mixed_events,
        "exact_cross_task_opposing_pairs": len(exact_pairs),
        "exact_pair_task_coverage": len(exact_tasks),
        "empirical_exact_signature_conditional_bayes_error": conditional_errors
        / max(len(ordered_ids), 1),
        "approximate_similarity": "fixed public hashing-TFIDF word+char4 cosine",
        "approximate_similarity_threshold": similarity_threshold,
        "approximate_cross_task_mutual_nearest_pairs": len(approximate_pairs),
        "approximate_pair_task_coverage": len(approximate_tasks),
        "approximate_min_similarity": min(
            (float(row["similarity"]) for row in approximate_pairs), default=None
        ),
    }
    return summary, pair_rows


def deterministic_split(task_id_hash: str) -> str:
    bucket = int(hashlib.sha256(task_id_hash.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "development"
    return "test"


def validate_benchmark_package_data(
    public_rows: Sequence[Mapping[str, Any]],
    private_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, bool]:
    public_ids = [str(row.get("event_id", "")) for row in public_rows]
    private_ids = [str(row.get("event_id", "")) for row in private_rows]
    split_ids = [str(row.get("event_id", "")) for row in split_rows]
    public_by_id = {str(row.get("event_id", "")): row for row in public_rows}
    task_splits: dict[str, str] = {}
    task_split_valid = True
    row_alignment = True
    for row in split_rows:
        event_id = str(row.get("event_id", ""))
        task = str(row.get("task_id_hash", ""))
        split = str(row.get("split", ""))
        public = public_by_id.get(event_id, {})
        row_alignment = row_alignment and str(public.get("task_id_hash", "")) == task
        if task in task_splits and task_splits[task] != split:
            task_split_valid = False
        task_splits[task] = split
        task_split_valid = task_split_valid and split == deterministic_split(task)
    private_alignment = all(
        str(row.get("task_id_hash", ""))
        == str(public_by_id.get(str(row.get("event_id", "")), {}).get("task_id_hash", ""))
        for row in private_rows
    )
    split_event_counts = dict(
        sorted(Counter(str(row.get("split", "")) for row in split_rows).items())
    )
    split_task_counts = {
        split: len({task for task, assigned in task_splits.items() if assigned == split})
        for split in ("train", "development", "test")
    }
    label_counts = dict(
        sorted(Counter(str(row.get("decision", "")) for row in private_rows).items())
    )
    source_names = [str(row.get("name", "")) for row in manifest.get("sources", [])]
    return {
        "manifest_status": manifest.get("status") == "completed"
        and manifest.get("version") == "compensation_trap_benchmark_v1",
        "official_ground_truth_manifest": manifest.get(
            "official_branch_evidence_recomputed"
        )
        is True
        and manifest.get("credit_mode") == "lexicographic_v4"
        and isinstance(manifest.get("horizon"), int)
        and manifest.get("horizon", 0) > 0
        and isinstance(manifest.get("atol"), (int, float))
        and manifest.get("atol", -1) >= 0,
        "event_identity": bool(public_ids)
        and len(public_ids) == len(set(public_ids))
        and public_ids == private_ids == split_ids,
        "field_boundary": all(set(row) == set(PUBLIC_FIELDS) for row in public_rows)
        and all(set(row) == set(PRIVATE_FIELDS) for row in private_rows),
        "label_boundary": all(str(row.get("decision", "")) in LABELS for row in private_rows),
        "public_private_task_alignment": private_alignment and row_alignment,
        "scenario_task_identity": all(
            str(row.get("task_id_hash", ""))
            == sha256_text(str(row.get("scenario_name", "")))
            for row in public_rows
        ),
        "task_disjoint_deterministic_splits": task_split_valid,
        "schema_bound": schema.get("version") == "compensation_trap_benchmark_v1"
        and schema.get("public_fields") == list(PUBLIC_FIELDS)
        and schema.get("private_fields") == list(PRIVATE_FIELDS)
        and schema.get("labels") == list(LABELS)
        and schema.get("split_rule")
        == "sha256(task_id_hash) prefix modulo 10: 0-5 train, 6-7 development, 8-9 test",
        "source_inventory": bool(source_names)
        and len(source_names) == len(set(source_names))
        and {str(row.get("source", "")) for row in public_rows} <= set(source_names),
        "statistics_recomputed": manifest.get("events") == len(public_rows)
        and manifest.get("tasks") == len(task_splits)
        and manifest.get("label_counts") == label_counts
        and manifest.get("split_event_counts") == split_event_counts
        and manifest.get("split_task_counts") == split_task_counts,
    }
