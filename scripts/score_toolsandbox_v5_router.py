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
from rescuecredit.toolsandbox_router import (
    apply_flip,
    build_router_features,
    completion_stats,
    router_probabilities,
)
from scripts.freeze_toolsandbox_v5_protocol import CONFIG, PROTOCOL_STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mask-adapter", type=Path, required=True)
    parser.add_argument("--control-router", type=Path, required=True)
    parser.add_argument("--v5-router", type=Path, required=True)
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
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("config") != CONFIG:
        raise ValueError("invalid V5 scoring protocol")
    if directory_sha256(args.model) != protocol.get("base_model_sha256"):
        raise ValueError("V5 scoring base model mismatch")
    if directory_sha256(args.mask_adapter) != protocol.get("mask_adapter_sha256"):
        raise ValueError("V5 scoring Mask adapter mismatch")
    source_ok = all(
        Path(path).is_file() and file_sha256(Path(path)) == digest
        for path, digest in protocol.get("source_sha256", {}).items()
    )
    if not source_ok:
        raise ValueError("V5 scoring source identity changed")
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
        raise ValueError("V5 scoring split manifest identity mismatch")
    role_manifest = json.loads(role_manifest_path.read_text(encoding="utf-8"))
    if role_manifest.get("public_sha256") != file_sha256(args.public_events):
        raise ValueError("V5 public events do not match the frozen scoring role")
    control = torch.load(args.control_router, map_location="cpu", weights_only=True)
    v5 = torch.load(args.v5_router, map_location="cpu", weights_only=True)
    if (
        control.get("method") != "margin_control"
        or v5.get("method") != "causal_router_v5"
    ):
        raise ValueError("V5 router method identity mismatch")
    if not (
        control.get("protocol_lock_sha256")
        == v5.get("protocol_lock_sha256")
        == file_sha256(args.protocol_lock)
    ):
        raise ValueError("V5 router protocol identity mismatch")

    public_rows = read_jsonl(args.public_events)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if CONFIG["fp32"] else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).cuda()
    model = PeftModel.from_pretrained(base, args.mask_adapter).eval()
    device = next(model.parameters()).device
    predictions = {
        method: [] for method in ("mask", "margin_control", "causal_router_v5")
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
                CONFIG["max_length"],
                device,
            )
            logp_b, hidden_b, tokens_b = completion_stats(
                model,
                tokenizer,
                prompt,
                canonical_completion(row["action_b"]),
                CONFIG["max_length"],
                device,
            )
            margin = float((logp_b - logp_a).detach())
            mask_selected = "b" if margin > 0 else "a"
            control_features, semantic_features = build_router_features(
                hidden_a,
                hidden_b,
                margin_b_over_a=margin,
                action_a_tokens=tokens_a,
                action_b_tokens=tokens_b,
                projection_dim=CONFIG["projection_dim"],
                projection_seed=CONFIG["projection_seed"],
            )
            method_features = {
                "margin_control": control_features.unsqueeze(0),
                "causal_router_v5": semantic_features.unsqueeze(0),
            }
            base_record = {
                "event_id": str(row["event_id"]),
                "task_id_hash": str(row["task_id_hash"]),
                "mask_selected": mask_selected,
                "margin_b_over_a": margin,
            }
            predictions["mask"].append(
                {**base_record, "selected": mask_selected, "flipped": False}
            )
            for method, checkpoint in (
                ("margin_control", control),
                ("causal_router_v5", v5),
            ):
                probability = float(
                    router_probabilities(method_features[method], checkpoint)[0]
                )
                flipped = probability >= float(checkpoint["threshold"])
                predictions[method].append(
                    {
                        **base_record,
                        "selected": apply_flip(mask_selected, flipped),
                        "flipped": flipped,
                        "flip_probability": probability,
                        "flip_threshold": float(checkpoint["threshold"]),
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
        "stage": "toolsandbox_v5_public_only_router_scoring",
        "evaluation_role": args.evaluation_role,
        "events": len(public_rows),
        "event_set_hash": event_set_hash(public_rows),
        "public_events_sha256": file_sha256(args.public_events),
        "split_manifest_sha256": file_sha256(role_manifest_path),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "mask_adapter_sha256": directory_sha256(args.mask_adapter),
        "control_router_sha256": file_sha256(args.control_router),
        "v5_router_sha256": file_sha256(args.v5_router),
        "prediction_sha256": prediction_hashes,
        "private_outcomes_read": False,
        "model_inputs": [
            "visible prompt",
            "public schemas",
            "candidate A",
            "candidate B",
        ],
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "scoring_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
