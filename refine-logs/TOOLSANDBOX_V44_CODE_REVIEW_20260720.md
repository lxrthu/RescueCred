# ToolSandbox V4.4 local implementation review

Date: 2026-07-20

Verdict: **DEPLOY SIGNAL AUDIT ONLY**.

## Blocking-risk checklist

- Candidate generation is reference-free: the worker receives only visible
  history, public Tool-ID schemas, proposal A, and the requested count.
- Both actions are public-schema complete before paired execution.  Copies,
  duplicates, and unsupported invented argument values are rejected.
- Candidate order is not treated as a label.  Official outcomes are read only
  after A and B are fixed and are stripped from the training artifact.
- The same offset-85 tasks used for prior training are reused deliberately.
  Offset 125 and offset 165 are only identity-bound and never executed.
- The failed V4.3 gate is hash-bound and preserved; no threshold is lowered.
- Snapshot and common-prefix equality are measured for every paired branch.
- A three-scenario sanity must pass before the full 40-scenario audit.
- The runner contains no adapter training.  A later learner protocol is
  authorized only by a passed V4.4 data gate.

## Frozen success bar

At least 60 replay-valid nonzero pairs, at least 8 rescue and 8 reverse pairs,
at least 5 tasks in each direction, no more than 4 pairs per task, maximum task
share 0.10, exact snapshots, and worker-failure rate at most 0.10.

## Compatibility note

The shared worker gained a new `diversify` mode.  Existing propose/repair modes
and their output validators remain regression tested, but the worker source
hash changes.  V4.4 therefore creates new source-bound locks and never claims
to replay old source identities.

## Local validation

- Python compile: pass.
- Ruff: pass.
- Focused ToolSandbox/V4.3/V4.4 tests: pass.
- Bash syntax: pass.
- Full repository pytest: pass with the pre-existing skipped tests only.

No V4.4 outcome has been observed locally.
