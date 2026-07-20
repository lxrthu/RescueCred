#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescuecredit.frozen_bank import (
    directory_sha256,
    file_sha256,
    read_jsonl,
    write_jsonl,
)
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import canonical_completion, event_set_hash
from rescuecredit.toolsandbox_router import build_router_features, completion_stats
from rescuecredit.toolsandbox_selective_router import (
    apply_platt_scaler,
    conservative_choice,
    probe_probabilities,
)
from scripts.freeze_toolsandbox_v6_protocol import PROTOCOL_STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mask-adapter", type=Path, required=True)
    parser.add_argument("--margin-probe", type=Path, required=True)
    parser.add_argument("--semantic-probe", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--public-events", type=Path, required=True)
    parser.add_argument(
        "--evaluation-role", choices=("development", "posthoc"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("invalid V6 scoring protocol")
    if directory_sha256(args.model) != protocol.get("base_model_sha256"):
        raise ValueError("V6 scoring base model mismatch")
    if directory_sha256(args.mask_adapter) != protocol.get("mask_adapter_sha256"):
        raise ValueError("V6 scoring Mask adapter mismatch")
    source_ok = all(
        Path(path).is_file() and file_sha256(Path(path)) == digest
        for path, digest in protocol.get("source_sha256", {}).items()
    )
    if not source_ok:
        raise ValueError("V6 scoring source identity changed")
    role_lock = (
        protocol["development"]
        if args.evaluation_role == "development"
        else protocol["posthoc_confirmation"]
    )
    role_manifest_path = Path(role_lock["data_dir"]) / "manifest.json"
    if (
        not role_manifest_path.is_file()
        or file_sha256(role_manifest_path) != role_lock["manifest_sha256"]
    ):
        raise ValueError("V6 scoring split manifest identity mismatch")
    role_manifest = json.loads(role_manifest_path.read_text(encoding="utf-8"))
    if role_manifest.get("public_sha256") != file_sha256(args.public_events):
        raise ValueError("V6 public events do not match the frozen role")

    margin_probe = torch.load(args.margin_probe, map_location="cpu", weights_only=True)
    semantic_probe = torch.load(
        args.semantic_probe, map_location="cpu", weights_only=True
    )
    expected_lock = file_sha256(args.protocol_lock)
    if (
        margin_probe.get("method") != "margin_probe"
        or semantic_probe.get("method") != "semantic_probe"
        or margin_probe.get("protocol_lock_sha256") != expected_lock
        or semantic_probe.get("protocol_lock_sha256") != expected_lock
    ):
        raise ValueError("V6 probe identity mismatch")

    public_rows = read_jsonl(args.public_events)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    feature_config = protocol["feature_config"]
    dtype = torch.float32 if feature_config["fp32"] else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).cuda()
    model = PeftModel.from_pretrained(base, args.mask_adapter).eval()
    device = next(model.parameters()).device
    predictions = {
        method: [] for method in ("default_b", "margin_probe", "semantic_probe")
    }
    started = time.time()
    with torch.no_grad():
        for index, row in enumerate(public_rows, start=1):
            prompt = str(row["prompt"])
            logp_a, hidden_a, tokens_a = completion_stats(
                model,
                tokenizer,
                prompt,
                canonical_completion(row["action_a"]),
                feature_config["max_length"],
                device,
            )
            logp_b, hidden_b, tokens_b = completion_stats(
                model,
                tokenizer,
                prompt,
                canonical_completion(row["action_b"]),
                feature_config["max_length"],
                device,
            )
            margin = float((logp_b - logp_a).detach())
            control_features, semantic_features = build_router_features(
                hidden_a,
                hidden_b,
                margin_b_over_a=margin,
                action_a_tokens=tokens_a,
                action_b_tokens=tokens_b,
                projection_dim=feature_config["projection_dim"],
                projection_seed=feature_config["projection_seed"],
            )
            base_record = {
                "event_id": str(row["event_id"]),
                "task_id_hash": str(row["task_id_hash"]),
                "margin_b_over_a": margin,
            }
            predictions["default_b"].append(
                {
                    **base_record,
                    "selected": "b",
                    "abstained_to_b": True,
                    "routed_to_a": False,
                }
            )
            for method, checkpoint, features in (
                ("margin_probe", margin_probe, control_features),
                ("semantic_probe", semantic_probe, semantic_features),
            ):
                raw = probe_probabilities(features.unsqueeze(0), checkpoint)
                probability = float(
                    apply_platt_scaler(raw, checkpoint["calibration"])[0]
                )
                threshold = float(checkpoint["threshold"])
                selected = conservative_choice(probability, threshold)
                predictions[method].append(
                    {
                        **base_record,
                        "selected": selected,
                        "abstained_to_b": selected == "b",
                        "routed_to_a": selected == "a",
                        "reverse_probability": probability,
                        "reverse_threshold": threshold,
                    }
                )
            if index % 20 == 0 or index == len(public_rows):
                print(json.dumps({"progress": f"{index}/{len(public_rows)}"}))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_hashes = {}
    for method, rows in predictions.items():
        path = args.output_dir / f"{method}.predictions.jsonl"
        write_jsonl(path, rows)
        prediction_hashes[method] = file_sha256(path)
    summary = {
        "status": "completed",
        "stage": "toolsandbox_v6_public_only_reverse_scoring",
        "evaluation_role": args.evaluation_role,
        "events": len(public_rows),
        "event_set_hash": event_set_hash(public_rows),
        "public_events_sha256": file_sha256(args.public_events),
        "split_manifest_sha256": file_sha256(role_manifest_path),
        "protocol_lock_sha256": expected_lock,
        "mask_adapter_sha256": directory_sha256(args.mask_adapter),
        "margin_probe_sha256": file_sha256(args.margin_probe),
        "semantic_probe_sha256": file_sha256(args.semantic_probe),
        "prediction_sha256": prediction_hashes,
        "private_outcomes_read": False,
        "model_inputs": [
            "visible prompt",
            "public schemas",
            "candidate A",
            "Harness correction B",
            "frozen policy representations and margin",
        ],
        "default_action": "b",
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "scoring_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
