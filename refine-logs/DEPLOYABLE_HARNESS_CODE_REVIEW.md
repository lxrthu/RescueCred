# Deployable Harness Stage 1 Code Review

**Review mode**: local-only
**Date**: 2026-07-15

## Scope

This review covers only reference isolation, the conservative deployable harness, its tri-state semantic validator, and the offline harness quality gate. RescueCredit-v2 losses were intentionally not implemented.

## Blocking findings and resolution

1. The historical Harness received `expected_action` and copied it into the correction. It is now named `OracleAPIBankHarness` and retained only as a diagnostic upper bound.
2. `DeployableAPIBankHarness` has no `expected` or `reference_actions` argument. It receives a filtered public observation, proposal, and optional prior tool receipt.
3. Schema-valid is not treated as semantic-valid. The deployable validator returns `true`, `false`, or `unknown` and trains nothing from `unknown` cases.
4. Existing argument values are not rewritten from goal text alone; this removed observed multi-intent password/appointment corruption.
5. Reference actions are used only inside the offline evaluator for case construction and scoring.

## Verification

- Full test suite: 41 passed.
- Ruff: passed on all changed Python files.
- Compileall: passed.
- Dev audit: 19 tasks, 37 clean cases, 183 corrupt cases.
- Corrections: 17/17 correct.
- Correction precision: 100%.
- Clean-action harm rate: 0%.
- Coverage: 9.29%.
- Single-step rescue rate: 9.29%.

## Verdict

**BLOCKED before V2 training.** Safety is adequate, but the predeclared 10% coverage and rescue thresholds were not met. A frozen-Qwen missing-argument generator has now been implemented behind the same strict validator; its GPU quality gate is pending. Do not implement the causal preference loss unless that gate passes.
