from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from rescuecredit.deltaguard_observers import (
    build_observer_plan,
    plan_family,
    plan_structure_payload,
)
from rescuecredit.deltaguard_probe import acquisition_selected, action_hash
from rescuecredit.deltaguard_toolsandbox import public_structure_digest
from rescuecredit.frozen_bank import read_jsonl


PROTOCOL_STATUS = "frozen_before_toolsandbox_deltaguard_collection"
PUBLIC_HMAC_KEY = "RescueCredit/DeltaGuard/public-acquisition/v1"
FULL_CONFIG = {
    "families": ["messaging", "reminders", "settings"],
    "source_events_per_family": 80,
    "attempt_cap_per_family": 120,
    "acquisition_rate": 0.25,
    "max_probe_rate": 0.30,
    "min_class_per_family": 6,
    "min_typed_delta_roc_auc": 0.75,
    "min_auc_gain_over_v7": 0.10,
    "risk_alpha": 0.05,
}


PUBLIC_EVENT_FIELDS = {
    "event_id",
    "task_id_hash",
    "scenario_name",
    "action_a",
    "action_b",
    "treatment_public_tool_schemas",
    "treatment_visible_history",
}


def export_public_event(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = PUBLIC_EVENT_FIELDS - set(row)
    if missing:
        raise ValueError(f"raw event lacks public fields: {sorted(missing)}")
    return {key: row[key] for key in sorted(PUBLIC_EVENT_FIELDS)}


def load_public_sources(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            extra = set(row) - PUBLIC_EVENT_FIELDS
            if extra:
                raise ValueError(
                    f"DeltaGuard public bank contains non-public fields: {sorted(extra)}"
                )
            event_id = str(row.get("event_id", ""))
            if not event_id or event_id in seen:
                raise ValueError(f"missing or duplicate source event id: {event_id}")
            seen.add(event_id)
            rows.append(dict(row))
    return rows


def visible_instruction(history: Sequence[Mapping[str, Any]]) -> str:
    user = [
        str(row.get("content", ""))
        for row in history
        if str(row.get("sender", "")).endswith("USER")
    ]
    return user[-1] if user else ""


def public_event_projection(row: Mapping[str, Any]) -> dict[str, Any] | None:
    event_id = row.get("event_id")
    action_a = row.get("action_a")
    action_b = row.get("action_b")
    schemas = row.get("treatment_public_tool_schemas")
    history = row.get("treatment_visible_history")
    if (
        not isinstance(event_id, str)
        or not isinstance(action_a, Mapping)
        or not isinstance(action_b, Mapping)
        or not isinstance(schemas, list)
        or not isinstance(history, list)
    ):
        return None
    if action_a == action_b:
        return None
    plan = build_observer_plan(
        action_a=action_a,
        action_b=action_b,
        schemas=[schema for schema in schemas if isinstance(schema, Mapping)],
        instruction=visible_instruction(
            [item for item in history if isinstance(item, Mapping)]
        ),
    )
    family = plan_family(plan)
    if family is None:
        return None
    return {
        "event_id": event_id,
        "task_id_hash": str(row.get("task_id_hash", "")),
        "scenario_name": str(row.get("scenario_name", "")),
        "family": family,
        "action_hash_a": action_hash(action_a),
        "action_hash_b": action_hash(action_b),
        "action_structure_a": public_structure_digest(action_a),
        "action_structure_b": public_structure_digest(action_b),
        "plan_structure": public_structure_digest(plan_structure_payload(plan)),
        "plan_predicates": len(plan),
    }


def freeze_source_stream(
    rows: Sequence[Mapping[str, Any]],
    *,
    families: Sequence[str],
    source_events_per_family: int,
    attempt_cap_per_family: int,
    acquisition_rate: float,
    hmac_key: str = PUBLIC_HMAC_KEY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source_events_per_family <= 0 or attempt_cap_per_family < source_events_per_family:
        raise ValueError("invalid source-stream counts")
    wanted = tuple(dict.fromkeys(str(family) for family in families))
    if not wanted:
        raise ValueError("at least one family is required")
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid = Counter()
    for row in rows:
        projection = public_event_projection(row)
        if projection is None:
            invalid["unsupported_public_projection"] += 1
            continue
        family = str(projection["family"])
        if family not in wanted:
            invalid["other_family"] += 1
            continue
        candidates[family].append(projection)
    selected: list[dict[str, Any]] = []
    counts = {}
    for family in wanted:
        ordered = sorted(
            candidates.get(family, []),
            key=lambda row: hashlib.sha256(
                ("DeltaGuard/source-order/v1:" + str(row["event_id"])).encode("utf-8")
            ).hexdigest(),
        )[:attempt_cap_per_family]
        chosen = ordered[:source_events_per_family]
        counts[family] = {"eligible_within_cap": len(ordered), "selected_source": len(chosen)}
        for row in chosen:
            item = dict(row)
            item["eligible"] = True
            item["selected"] = acquisition_selected(
                event_id=str(row["event_id"]),
                eligible=True,
                key=hmac_key,
                rate=acquisition_rate,
            )
            selected.append(item)
    complete = all(
        counts[family]["selected_source"] == source_events_per_family for family in wanted
    )
    selected.sort(key=lambda row: (str(row["family"]), str(row["event_id"])))
    audit = {
        "complete": complete,
        "families": list(wanted),
        "counts": counts,
        "invalid": dict(invalid),
        "source_events": len(selected),
        "probe_events": sum(bool(row["selected"]) for row in selected),
        "realized_frozen_probe_rate": (
            sum(bool(row["selected"]) for row in selected) / len(selected)
            if selected
            else 0.0
        ),
        "labels_inspected": False,
    }
    return selected, audit


def source_stream_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(rows), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_for_role(
    role: str,
    *,
    families: Sequence[str] | None = None,
    source_events_per_family: int | None = None,
    attempt_cap_per_family: int | None = None,
) -> dict[str, Any]:
    if role == "full":
        config = dict(FULL_CONFIG)
        if families is not None and list(families) != FULL_CONFIG["families"]:
            raise ValueError("full role uses the preregistered three families")
        if source_events_per_family not in (None, FULL_CONFIG["source_events_per_family"]):
            raise ValueError("full role uses 80 source events per family")
        if attempt_cap_per_family not in (None, FULL_CONFIG["attempt_cap_per_family"]):
            raise ValueError("full role uses the frozen attempt cap")
        return config
    defaults = {
        "sanity": (2, 12, 1),
        "feasibility": (10, 40, 2),
    }
    if role not in defaults:
        raise ValueError(f"unknown DeltaGuard role: {role}")
    count, cap, minimum = defaults[role]
    config = dict(FULL_CONFIG)
    config.update(
        {
            "families": list(families or FULL_CONFIG["families"]),
            "source_events_per_family": source_events_per_family or count,
            "attempt_cap_per_family": attempt_cap_per_family or cap,
            "min_class_per_family": minimum,
        }
    )
    return config


def verify_protocol_source_identity(protocol: Mapping[str, Any]) -> None:
    from rescuecredit.frozen_bank import file_sha256

    for path, digest in protocol.get("source_sha256", {}).items():
        source = Path(path)
        if not source.is_file() or file_sha256(source) != digest:
            raise ValueError(f"DeltaGuard source identity changed: {path}")
    for item in protocol.get("public_sources", []):
        source = Path(str(item["path"]))
        if not source.is_file() or file_sha256(source) != item["sha256"]:
            raise ValueError(f"DeltaGuard public source identity changed: {source}")
    bound = (
        ("public_bank_manifest", "public_bank_manifest_sha256"),
        ("v7_checkpoint", "v7_checkpoint_sha256"),
        ("v7_train_file", "v7_train_file_sha256"),
        ("v7_run_summary", "v7_run_summary_sha256"),
        ("v7_protocol_lock", "v7_protocol_lock_sha256"),
        ("v7_oof", "v7_oof_sha256"),
    )
    for path_key, hash_key in bound:
        raw_path = protocol.get(path_key)
        digest = protocol.get(hash_key)
        if raw_path is None and digest is None:
            continue
        path = Path(str(raw_path))
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"DeltaGuard bound artifact changed: {path_key}")
