from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "rescuecredit.route_a_bank.v1"
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "expected",
        "expected_action",
        "reference_action",
        "reference_actions",
        "correct_after",
        "offline_label",
        "ground_truth",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def event_id(
    *, task_id: str, call_index: int, parameter: str, proposal: dict[str, Any]
) -> str:
    return digest(
        {
            "task_id": task_id,
            "call_index": int(call_index),
            "parameter": parameter,
            "proposal": proposal,
        }
    )


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def validate_public_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "event_id",
        "split",
        "task_id",
        "prompt",
        "action_a",
        "action_b",
        "patch_id",
        "provenance",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"bank record missing fields: {missing}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported bank schema: {record['schema_version']!r}")
    leaked = sorted(FORBIDDEN_PUBLIC_KEYS.intersection(_walk_keys(record)))
    if leaked:
        raise ValueError(f"protected offline fields leaked into public bank: {leaked}")
    if record["split"] != "train":
        raise ValueError("the frozen correction bank may contain train events only")
    if record["action_a"] == record["action_b"]:
        raise ValueError("action_a and action_b must differ")
    if len(str(record["event_id"])) != 64:
        raise ValueError("event_id must be a SHA-256 hex digest")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash a directory by sorted relative paths and per-file content hashes."""

    files = sorted(child for child in path.rglob("*") if child.is_file())
    return digest(
        [
            {
                "path": child.relative_to(path).as_posix(),
                "sha256": file_sha256(child),
            }
            for child in files
        ]
    )
