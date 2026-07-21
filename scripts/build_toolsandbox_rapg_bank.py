#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from rescuecredit.frozen_bank import (
    directory_sha256,
    file_sha256,
    read_jsonl,
    write_jsonl,
)
from rescuecredit.logging import write_json
from rescuecredit.rapg import public_hash_features, stable_seed
from rescuecredit.toolsandbox_preference import canonical_completion
from scripts.train_route_a_preference import mean_completion_logprob


def _projection(model) -> tuple[list[str], list[float], float]:
    names: list[str] = []
    values: list[float] = []
    squared_norm = 0.0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        names.append(name)
        if parameter.grad is None:
            values.append(0.0)
            continue
        gradient = parameter.grad.detach().float()
        squared_norm += float(gradient.square().sum())
        values.append(float(gradient.sum() / math.sqrt(gradient.numel())))
    if not names or squared_norm <= 0:
        raise RuntimeError("LoRA score gradient is empty")
    return names, values, math.sqrt(squared_norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-events", type=Path, required=True)
    parser.add_argument("--executed-b-returns", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--hash-dimension", type=int, default=128)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if not args.model.is_dir() or not args.adapter.is_dir():
        raise FileNotFoundError("base model or adapter directory is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = args.output_dir / "rapg_bank.pt"
    ledger_path = args.output_dir / "behavior_ledger.jsonl"
    manifest_path = args.output_dir / "bank_manifest.json"
    if any(path.exists() for path in (bank_path, ledger_path, manifest_path)):
        raise FileExistsError("refusing to overwrite RAPG bank artifacts")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "completed":
        raise ValueError("RAPG source manifest is not complete")
    if source_manifest.get("public_sha256") != file_sha256(args.public_events):
        raise ValueError("RAPG public source identity mismatch")
    if source_manifest.get("executed_b_sha256") != file_sha256(
        args.executed_b_returns
    ):
        raise ValueError("RAPG executed-B source identity mismatch")
    if source_manifest.get("outcome_direction_filter_used") is not False:
        raise ValueError("RAPG source was selected using outcome direction")
    public_rows = read_jsonl(args.public_events)
    if len(public_rows) < 20:
        raise ValueError("RAPG Pilot 0 requires at least 20 public events")
    event_ids = [str(row["event_id"]) for row in public_rows]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("public event ids are not unique")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).cuda()
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, args.adapter, is_trainable=True
    ).eval()
    device = next(model.parameters()).device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("adapter exposes no trainable parameters for score extraction")

    started = time.time()
    ledger: list[dict[str, Any]] = []
    score_sketches = []
    score_norms = []
    candidate_scores = []
    action_probabilities = []
    proposal_indices = []
    task_ids = []
    projection_names: list[str] | None = None
    public_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(public_rows, start=1):
        event_id = str(row["event_id"])
        prompt = str(row["prompt"])
        action_a = row["action_a"]
        action_b = row["action_b"]
        if action_a == action_b:
            raise ValueError(f"candidate actions are identical: {event_id}")
        model.zero_grad(set_to_none=True)
        score_a = mean_completion_logprob(
            model,
            tokenizer,
            prompt,
            canonical_completion(action_a),
            args.max_length,
            device,
        )
        score_b = mean_completion_logprob(
            model,
            tokenizer,
            prompt,
            canonical_completion(action_b),
            args.max_length,
            device,
        )
        scores = torch.stack([score_a, score_b]).float()
        log_probabilities = torch.log_softmax(scores / args.temperature, dim=0)
        probabilities = log_probabilities.detach().exp().cpu()
        generator = torch.Generator(device="cpu").manual_seed(
            stable_seed(args.seed, event_id)
        )
        proposal_index = int(
            torch.multinomial(probabilities, 1, generator=generator).item()
        )
        log_probabilities[proposal_index].backward()
        names, sketch, exact_norm = _projection(model)
        if projection_names is None:
            projection_names = names
        elif names != projection_names:
            raise RuntimeError("LoRA score projection changed across events")
        score_sketches.append(sketch)
        score_norms.append(exact_norm)
        candidate_scores.append([float(score_a.detach()), float(score_b.detach())])
        action_probabilities.append([float(value) for value in probabilities])
        proposal_indices.append(proposal_index)
        task_ids.append(str(row["task_id_hash"]))
        public_by_id[event_id] = row
        ledger.append(
            {
                "event_id": event_id,
                "task_id_hash": str(row["task_id_hash"]),
                "candidate_policy": "softmax_mean_sequence_logprob_v1",
                "candidate_scores": candidate_scores[-1],
                "action_probabilities": action_probabilities[-1],
                "proposal_index": proposal_index,
                "proposal_action": action_a if proposal_index == 0 else action_b,
                "executed_default_index": 1,
                "sampling_seed": stable_seed(args.seed, event_id),
                "private_outcome_opened": False,
            }
        )
        if index % 10 == 0 or index == len(public_rows):
            print(json.dumps({"progress": f"{index}/{len(public_rows)}"}), flush=True)

    # Seal the actual behavior samples before opening any Full-Shadow outcome.
    write_jsonl(ledger_path, ledger)
    behavior_ledger_sha256 = file_sha256(ledger_path)

    executed_rows = read_jsonl(args.executed_b_returns)
    executed_by_id = {str(row["event_id"]): row for row in executed_rows}
    if set(executed_by_id) != set(public_by_id):
        raise ValueError("public and executed-B RAPG event sets differ")
    executed_returns = []
    features = []
    for event_id, proposal_index, scores, probabilities in zip(
        event_ids,
        proposal_indices,
        candidate_scores,
        action_probabilities,
        strict=True,
    ):
        public = public_by_id[event_id]
        y_b = float(executed_by_id[event_id]["executed_b_return"])
        if not math.isfinite(y_b):
            raise ValueError("non-finite executed B return")
        hashed = public_hash_features(
            [
                public["prompt"],
                public["action_a"],
                public["action_b"],
                public["action_a"] if proposal_index == 0 else public["action_b"],
            ],
            dimension=args.hash_dimension,
        )
        numeric = [
            float(scores[0]),
            float(scores[1]),
            float(scores[1] - scores[0]),
            float(probabilities[0]),
            float(probabilities[1]),
            float(proposal_index),
            y_b,
        ]
        features.append(hashed + numeric)
        executed_returns.append(y_b)

    bank = {
        "version": "rapg_candidate_policy_bank_v1",
        "event_ids": event_ids,
        "task_ids": task_ids,
        "proposal_indices": torch.tensor(proposal_indices, dtype=torch.long),
        "replaced": torch.tensor(
            [index == 0 for index in proposal_indices], dtype=torch.bool
        ),
        "action_probabilities": torch.tensor(
            action_probabilities, dtype=torch.float32
        ),
        "candidate_scores": torch.tensor(candidate_scores, dtype=torch.float32),
        "score_sketches": torch.tensor(score_sketches, dtype=torch.float32),
        "score_norms": torch.tensor(score_norms, dtype=torch.float32),
        "public_features": torch.tensor(features, dtype=torch.float32),
        "executed_returns": torch.tensor(executed_returns, dtype=torch.float32),
        "projection_names": projection_names or [],
        "behavior_ledger_sha256": behavior_ledger_sha256,
    }
    torch.save(bank, bank_path)
    base_model_sha256 = directory_sha256(args.model)
    adapter_sha256 = directory_sha256(args.adapter)
    manifest = {
        "status": "completed",
        "stage": "toolsandbox_rapg_candidate_policy_bank",
        "events": len(event_ids),
        "tasks": len(set(task_ids)),
        "replacement_events": sum(index == 0 for index in proposal_indices),
        "nonreplacement_events": sum(index == 1 for index in proposal_indices),
        "candidate_policy": "softmax_mean_sequence_logprob_v1",
        "temperature": args.temperature,
        "seed": args.seed,
        "max_length": args.max_length,
        "hash_dimension": args.hash_dimension,
        "score_projection": "registered_per_lora_tensor_normalized_sum_v1",
        "score_projection_dimension": len(projection_names or []),
        "exact_lora_score_norm_stored": True,
        "behavior_sampled_before_private_outcomes": True,
        "behavior_ledger_sha256": behavior_ledger_sha256,
        "bank_sha256": file_sha256(bank_path),
        "public_events_sha256": file_sha256(args.public_events),
        "executed_b_returns_sha256": file_sha256(args.executed_b_returns),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "base_model": str(args.model.resolve()),
        "base_model_sha256": base_model_sha256,
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": adapter_sha256,
        "ground_truth": source_manifest["ground_truth"],
        "old_deepseek_proposal_reused_as_on_policy_sample": False,
        "shadow_a_outcome_loaded_by_builder": False,
        "primary_propensity_clipping": False,
        "wall_time_sec": time.time() - started,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
