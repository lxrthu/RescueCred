import json
from pathlib import Path

from rescuecredit.frozen_bank import directory_sha256
from scripts.check_route_a_v3_gate import EXPECTED_CONFIG


def test_directory_sha256_is_content_and_path_bound(tmp_path: Path):
    root = tmp_path / "adapter"
    root.mkdir()
    (root / "a.json").write_text("one", encoding="utf-8")
    first = directory_sha256(root)
    assert first == directory_sha256(root)
    (root / "a.json").write_text("two", encoding="utf-8")
    assert first != directory_sha256(root)
    (root / "b.json").write_text("one", encoding="utf-8")
    assert first != directory_sha256(root)


def test_v3_expected_config_is_frozen():
    assert EXPECTED_CONFIG == {
        "method": "v3",
        "seed": 42,
        "epochs": 3,
        "learning_rate": 3e-6,
        "gradient_accumulation": 8,
        "max_length": 2048,
        "beta": 1.0,
        "max_causal_weight": 2.5,
        "v2_presentations_per_epoch": 0,
        "absolute_margin_coef": 1.0,
        "target_margin": 0.05,
        "lora_r": 16,
        "lora_alpha": 32,
        "fp32": True,
    }
