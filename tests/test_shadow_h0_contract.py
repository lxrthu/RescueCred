from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_shadow_continuation_never_reenables_harness():
    text = (ROOT / "scripts/run_train.py").read_text(encoding="utf-8")
    shadow = text.split("    def shadow_factory():", 1)[1].split("    if args.method ==", 1)[0]
    assert "execute_harness_action(" not in shadow
    assert '"harness_patch": None' in shadow
    assert "return proposal" in shadow


def test_full_shadow_evaluation_uses_unassisted_continuation():
    text = (ROOT / "scripts/evaluate_full_shadow.py").read_text(encoding="utf-8")
    continuation = text.split("    def continuation(", 1)[1].split("    shadow =", 1)[0]
    assert "shadow_harness.execute" not in continuation
    assert "return proposal" in continuation
