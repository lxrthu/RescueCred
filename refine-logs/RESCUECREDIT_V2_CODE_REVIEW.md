# RescueCredit-v2 Code Review

**Review mode**: local-only
**Date**: 2026-07-15

## Implemented contract

- Historical Oracle Mask and RescueCredit-v1 remain available for diagnosis and ablation.
- `mask_correction_v2` and `rescuecredit_v2` share the same reference-free Deployable Harness and frozen-Qwen repair generator.
- V2 masks the intervened prefix exactly like Mask; it never routes `g0_hat` into GRPO.
- Ordinary correction preferences are gated by joint tri-state A/B semantic validity.
- Causal preferences require an actual selected audit and observed replay-valid `shadow_return`.
- Causal weights use each event's committed probability and are clipped at 2.5.
- Action scores use mean action-token log-probability.
- A verified B with a negative trajectory delta is logged as `trajectory_conflict`; invalid A is not learned back.
- Fixed-main experiments run until at least the requested main-step target and report synchronized-batch overshoot; shadow work is reported separately.
- Deployable dispatch has a test that fails if `expected_action()` is queried.

## Verification

- Full local suite: 52 tests passed.
- Ruff: passed on project code.
- Compileall/py_compile: passed.
- `mask_correction_v2` dry-run: passed.
- `rescuecredit_v2` dry-run: passed.
- Deployable Harness GPU gate supplied by the user: passed with 95.45% correction precision, 12.02% coverage, 11.48% single-step rescue and 0% harm.

## Remaining blocking gate

The real two-GPU V2 smoke has not run yet. It must produce a completed summary, zero replay failures, at least one preference event and at least one non-zero causal loss before the 2000-main-step fair pilot is launched.
