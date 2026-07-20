from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rescuecredit.frozen_bank import read_jsonl
from rescuecredit.toolsandbox_active_shadow import build_active_shadow_features
from rescuecredit.toolsandbox_selective_router import probe_probabilities


def compute_v7_baseline_scores(
    *,
    probe_rows: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    hash_dimension: int,
    oof_path: Path | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Score receipts using frozen OOF values or a lineage-bound checkpoint.

    OOF values are preferred for events from the original V7 bank. Disjoint new
    events are scored by the frozen final V7 head. The caller audits which source
    was used for every event.
    """

    import torch

    oof = {}
    if oof_path is not None:
        rows = read_jsonl(oof_path)
        oof = {str(row["event_id"]): float(row["active_raw_score"]) for row in rows}
        if len(oof) != len(rows):
            raise ValueError("V7 OOF file contains duplicate event IDs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    head = checkpoint.get("active_head")
    if not isinstance(head, dict):
        raise ValueError("V7 checkpoint lacks active_head")

    scores: dict[str, float] = {}
    sources: dict[str, str] = {}
    pending_ids = []
    features = []
    for row in probe_rows:
        event_id = str(row["event_id"])
        if event_id in oof:
            scores[event_id] = oof[event_id]
            sources[event_id] = "frozen_v7_oof"
            continue
        evidence = row.get("evidence")
        if not isinstance(evidence, Mapping):
            scores[event_id] = 0.5
            sources[event_id] = "collection_error_default"
            continue
        feature_row = {
            "action_a": evidence["action_a"],
            "action_b": evidence["action_b"],
            "branch_a": {"receipts": [evidence["branch_a"]["action_receipt"]]},
            "branch_b": {"receipts": [evidence["branch_b"]["action_receipt"]]},
        }
        features.append(
            build_active_shadow_features(feature_row, hash_dimension=hash_dimension)
        )
        pending_ids.append(event_id)
    if features:
        values = probe_probabilities(torch.tensor(features, dtype=torch.float32), head)
        for event_id, value in zip(pending_ids, values, strict=True):
            scores[event_id] = float(value)
            sources[event_id] = "lineage_bound_frozen_v7_checkpoint"
    return scores, sources
