# ToolSandbox V4 Pool Repair Code Review

**Review mode:** local-only (secondary-agent delegation unavailable under the
active execution policy)

## Verdict

DEPLOY YES.

## Blocking checklist

- Freeze/evaluation selection parity: PASS. Both use the same
  `allow_distraction_tools=True` tiered pool for V4.
- Development/fresh leakage: PASS. The original no-distraction ordering is an
  exact prefix; protocol freezing hashes both sets and rejects intersections.
- Outcome-dependent resampling: PASS. Tier membership and within-tier shuffle
  depend only on public scenario categories and seed 42.
- Protocol drift: PASS. The named pool profile is centralized, stored in the
  lock, and validated at evaluation.
- Legacy audit behavior: PASS. Non-V4 selection retains the no-distraction-only
  default.
- Regression coverage: PASS. A synthetic 80-scenario test proves the old
  40-row prefix, a 40-row fresh slice, and zero overlap. Protocol-profile drift
  is rejected.

## Validation

- Directed ToolSandbox tests: 19 passed.
- Full repository tests: passed, with 2 intentional skips.
- Ruff on all touched Python files: passed.
- Git diff whitespace check: passed.
- V4 cloud runner Bash syntax: passed.

The full-repository Ruff command still reports unrelated pre-existing Route-A
violations; no reported violation is in a file modified by this repair.
