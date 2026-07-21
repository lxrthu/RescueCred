#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.compensation_trap import (
    build_collision_audit,
    validate_benchmark_package_data,
)
from rescuecredit.frozen_bank import file_sha256, read_jsonl, write_jsonl
from rescuecredit.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite collision audit")
    public_path = args.benchmark_dir / "public_events.jsonl"
    private_path = args.benchmark_dir / "private_outcomes.jsonl"
    split_path = args.benchmark_dir / "splits.jsonl"
    schema_path = args.benchmark_dir / "schema.json"
    card_path = args.benchmark_dir / "dataset_card.json"
    manifest_path = args.benchmark_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    public_rows = read_jsonl(public_path)
    private_rows = read_jsonl(private_path)
    split_rows = read_jsonl(split_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    card = json.loads(card_path.read_text(encoding="utf-8"))
    validation = validate_benchmark_package_data(
        public_rows, private_rows, split_rows, schema, manifest
    )
    integrity = {
        **validation,
        "file_hashes": manifest.get("public_sha256") == file_sha256(public_path)
        and manifest.get("private_sha256") == file_sha256(private_path)
        and manifest.get("splits_sha256") == file_sha256(split_path)
        and manifest.get("schema_sha256") == file_sha256(schema_path),
        "dataset_card_bound": manifest.get("dataset_card_sha256")
        == file_sha256(card_path)
        and card.get("release_authorized") is False
        and card.get("requires_upstream_license_review") is True,
        "deployable_feature_boundary": set(card.get("provenance_only_fields", []))
        == {
            "scenario_name",
            "reference_free_prefix_steps",
            "source",
            "task_id_hash",
        },
    }
    if not all(integrity.values()):
        raise ValueError({"collision_benchmark_integrity_failure": integrity})
    summary, pairs = build_collision_audit(
        public_rows,
        private_rows,
        similarity_threshold=args.similarity_threshold,
    )
    args.output_dir.mkdir(parents=True)
    pair_path = args.output_dir / "collision_pairs.private.jsonl"
    write_jsonl(pair_path, pairs)
    exact_pass = (
        summary["exact_cross_task_opposing_pairs"] >= 5
        and summary["exact_pair_task_coverage"] >= 3
    )
    approximate_pass = (
        summary["approximate_cross_task_mutual_nearest_pairs"] >= 20
        and summary["approximate_pair_task_coverage"] >= 3
        and summary["approximate_min_similarity"] is not None
        and summary["approximate_min_similarity"] >= args.similarity_threshold
    )
    passed = all(integrity.values()) and (exact_pass or approximate_pass)
    result = {
        "passed": passed,
        "exact_collision_claim_supported": exact_pass,
        "approximate_representation_claim_supported": approximate_pass,
        "information_theoretic_nonidentifiability_claim_supported": False,
        "metrics": summary,
        "checks": {
            **integrity,
            "exact_pair_gate": exact_pass,
            "approximate_pair_gate": approximate_pass,
        },
        "benchmark_manifest_sha256": file_sha256(manifest_path),
        "collision_pairs_sha256": file_sha256(pair_path),
        "claim_boundary": "collision evidence is conditional on the frozen public signature/TFIDF representation; it does not prove that all deployment-visible information is insufficient",
    }
    write_json(args.output_dir / "collision_gate.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
