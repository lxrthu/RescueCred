from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rescuecredit.frozen_bank import file_sha256, write_jsonl
from rescuecredit.logging import write_json
from rescuecredit.toolsandbox_preference import event_set_hash
from scripts.audit_toolsandbox_editcredit_gradients import _countsketch_coefficients
from scripts.freeze_toolsandbox_editcredit_protocol import STATUS


def _variance_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    source = tmp_path / "bound_source.py"
    source.write_text("# frozen\n", encoding="utf-8")
    train = tmp_path / "train.jsonl"
    rows = [
        {
            "event_id": f"event-{index}",
            "task_id_hash": f"task-{index // 2}",
            "decision": "rescue_preference" if index % 2 == 0 else "reverse_preference",
        }
        for index in range(4)
    ]
    write_jsonl(train, rows)
    protocol = tmp_path / "protocol.json"
    write_json(
        protocol,
        {
            "status": STATUS,
            "config": {"seed": 42, "gradient_accumulation": 2},
            "efficiency_config": {
                "gradient_sketch_buckets": 2,
                "gradient_bootstrap_batch_size": 2,
                "gradient_bootstrap_replicates": 100,
                "max_gradient_noise_scale_ratio": 0.7,
                "max_minibatch_gradient_mse_ratio": 0.7,
            },
            "train_sha256": file_sha256(train),
            "train_event_set_hash": event_set_hash(rows),
            "base_model_sha256": "base",
            "source_sha256": {str(source): file_sha256(source)},
        },
    )
    protocol_sha = file_sha256(protocol)
    paths = {"train": train}
    sketches = {
        "full_action": [[2.0, 0.0], [0.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
        "editcredit": [[1.1, 0.0], [0.9, 0.0], [1.1, 0.0], [0.9, 0.0]],
    }
    countsketch_hash = {
        "family": "independent_affine_mod_prime",
        "prime": 2_147_483_647,
        "coefficients": list(_countsketch_coefficients(42)),
    }
    for method, values in sketches.items():
        method_rows = [
            {
                **row,
                "method": method,
                "gradient_norm": abs(value[0]),
                "sketch": value,
            }
            for row, value in zip(rows, values, strict=True)
        ]
        sketch_path = tmp_path / f"{method}.jsonl"
        write_jsonl(sketch_path, method_rows)
        summary_path = tmp_path / f"{method}.summary.json"
        write_json(
            summary_path,
            {
                "status": "completed",
                "method": method,
                "protocol_lock_sha256": protocol_sha,
                "train_file_sha256": file_sha256(train),
                "base_model_sha256": "base",
                "sketches_sha256": file_sha256(sketch_path),
                "buckets": 2,
                "seed": 42,
                "event_set_hash": event_set_hash(method_rows),
                "source_sha256": {str(source): file_sha256(source)},
                "initial_trainable_sha256": "same-init",
                "countsketch_hash": countsketch_hash,
                "wall_time_sec": 1.0,
                "forward_calls": 16,
            },
        )
        paths[f"{method}_sketch"] = sketch_path
        paths[f"{method}_summary"] = summary_path
    return protocol, paths


def _run_variance_gate(protocol: Path, paths: dict[str, Path], output: Path):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_toolsandbox_editcredit_variance",
            "--protocol-lock",
            str(protocol),
            "--train-file",
            str(paths["train"]),
            "--full-summary",
            str(paths["full_action_summary"]),
            "--full-sketches",
            str(paths["full_action_sketch"]),
            "--edit-summary",
            str(paths["editcredit_summary"]),
            "--edit-sketches",
            str(paths["editcredit_sketch"]),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_variance_gate_passes_and_rejects_method_tamper(tmp_path: Path):
    protocol, paths = _variance_fixture(tmp_path)
    first_output = tmp_path / "variance.json"
    first = _run_variance_gate(protocol, paths, first_output)
    assert first.returncode == 0, first.stderr + first.stdout
    assert json.loads(first_output.read_text(encoding="utf-8"))["passed"] is True

    rows = [json.loads(line) for line in paths["editcredit_sketch"].read_text().splitlines()]
    rows[0]["method"] = "full_action"
    write_jsonl(paths["editcredit_sketch"], rows)
    summary = json.loads(paths["editcredit_summary"].read_text(encoding="utf-8"))
    summary["sketches_sha256"] = file_sha256(paths["editcredit_sketch"])
    write_json(paths["editcredit_summary"], summary)
    second_output = tmp_path / "variance_tampered.json"
    second = _run_variance_gate(protocol, paths, second_output)
    assert second.returncode == 1
    tampered = json.loads(second_output.read_text(encoding="utf-8"))
    assert tampered["integrity_checks"]["task_groups_rebuilt"] is False


def test_variance_gate_rejects_lora_initialization_mismatch(tmp_path: Path):
    protocol, paths = _variance_fixture(tmp_path)
    summary = json.loads(paths["editcredit_summary"].read_text(encoding="utf-8"))
    summary["initial_trainable_sha256"] = "different-init"
    write_json(paths["editcredit_summary"], summary)
    output = tmp_path / "variance_init_tamper.json"
    result = _run_variance_gate(protocol, paths, output)
    assert result.returncode == 1
    checked = json.loads(output.read_text(encoding="utf-8"))
    assert checked["integrity_checks"]["same_lora_initialization"] is False


def test_runner_does_not_expand_local_variables_before_assignment():
    runner = Path("scripts/cloud/run_toolsandbox_editcredit_seed42.sh").read_text(
        encoding="utf-8"
    )
    assert 'gpu="$2" directory="$OUT/gradient/$method"' not in runner
    assert 'gpu="$3" directory="$OUT/$method/fold$fold"' not in runner
