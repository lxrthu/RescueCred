# ToolSandbox V4.6 local code review

## Verdict

`DEPLOY_YES` for the development-only V4.6 engineering run. This does not
authorize a fresh-confirmation or paper claim.

## Reviewed invariants

- Both arms load the exact frozen V4.5 Mask adapter.
- Both arms consume all 126 frozen V4.4 events in the same deterministic order
  for three epochs and perform the same number of presentations.
- V4.6 routes already-correct signed Mask margins to retention and weak/wrong
  margins to signed causal correction.
- The matched control continues the Mask preference for B, separating the
  causal routing effect from additional training compute.
- Protocol, source, base-model, adapter, data, event-sequence, result, and raw
  metric identities are checked.
- Known V4.5 confirmation outcomes are explicitly post-hoc and cannot pass the
  V4.6 development gate.

## Validation

- Ruff: passed.
- Python compilation: passed.
- Focused ToolSandbox tests: 20 passed.
- Full repository test suite: passed, with two existing skips.
- `git diff --check`: passed.

## Remaining risk

The selective threshold and residual target are development choices informed
by the failed V4.5 result. Even a passing V4.6 development gate is only evidence
that the repaired learner behaves as intended on known fixtures. A new frozen
scenario profile is required before any confirmatory claim.
