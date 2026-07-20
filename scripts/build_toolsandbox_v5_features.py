#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256, file_sha256, read_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import canonical_completion, event_set_hash
from rescuecredit.toolsandbox_router import (
    build_router_features,
    completion_stats,
    desired_candidate,
    flip_target,
)
from scripts.freeze_toolsandbox_v5_protocol import CONFIG, PROTOCOL_STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mask-adapter", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("config") != CONFIG:
        raise ValueError("invalid V5 protocol")
    if file_sha256(args.train_file) != protocol.get("train_sha256"):
        raise ValueError("V5 training data identity mismatch")
    if directory_sha256(args.model) != protocol.get("base_model_sha256"):
        raise ValueError("V5 base model identity mismatch")
    if directory_sha256(args.mask_adapter) != protocol.get("mask_adapter_sha256"):
        raise ValueError("V5 Mask adapter identity mismatch")
    source_ok = all(
        Path(path).is_file() and file_sha256(Path(path)) == digest
        for path, digest in protocol.get("source_sha256", {}).items()
    )
    if not source_ok:
        raise ValueError("V5 source identity changed")

    rows = read_jsonl(args.train_file)
    if (
        len(rows) != protocol["train_events"]
        or event_set_hash(rows) != protocol["train_event_set_hash"]
    ):
        raise ValueError("V5 training event set mismatch")
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
    started = time.time()
    control_features = []
    semantic_features = []
    labels = []
    mask_selected = []
    decisions = []
    event_ids = []
    task_ids = []
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
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
            selected = "b" if margin > 0 else "a"
            control, semantic = build_router_features(
                hidden_a,
                hidden_b,
                margin_b_over_a=margin,
                action_a_tokens=tokens_a,
                action_b_tokens=tokens_b,
                projection_dim=CONFIG["projection_dim"],
                projection_seed=CONFIG["projection_seed"],
            )
            control_features.append(control)
            semantic_features.append(semantic)
            labels.append(flip_target(selected, str(row["decision"])))
            mask_selected.append(selected)
            decisions.append(str(row["decision"]))
            event_ids.append(str(row["event_id"]))
            task_ids.append(str(row["task_id_hash"]))
            if index % 20 == 0 or index == len(rows):
                print(json.dumps({"progress": f"{index}/{len(rows)}"}))

    payload = {
        "control_features": torch.stack(control_features),
        "semantic_features": torch.stack(semantic_features),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "mask_selected": mask_selected,
        "decisions": decisions,
        "event_ids": event_ids,
        "task_ids": task_ids,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "train_features.pt"
    torch.save(payload, feature_path)
    manifest = {
        "status": "completed",
        "stage": "toolsandbox_v5_frozen_mask_feature_cache",
        "events": len(rows),
        "flip_labels": int(sum(labels)),
        "keep_labels": len(labels) - int(sum(labels)),
        "mask_train_accuracy": sum(
            selected == desired_candidate(decision)
            for selected, decision in zip(mask_selected, decisions, strict=True)
        )
        / len(rows),
        "control_feature_dim": int(payload["control_features"].shape[1]),
        "semantic_feature_dim": int(payload["semantic_features"].shape[1]),
        "feature_file_sha256": file_sha256(feature_path),
        "train_sha256": file_sha256(args.train_file),
        "mask_adapter_sha256": directory_sha256(args.mask_adapter),
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "private_branch_outcomes_cached": False,
        "feature_inputs": [
            "visible prompt and public schemas",
            "candidate A and B",
            "frozen Mask completion representations",
            "frozen Mask margin",
        ],
        "wall_time_sec": time.time() - started,
    }
    write_json(args.output_dir / "feature_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
