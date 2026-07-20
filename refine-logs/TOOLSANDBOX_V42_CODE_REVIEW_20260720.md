# ToolSandbox V4.2 local implementation review

Date: 2026-07-20

Verdict: **DEPLOY YES** for the frozen seed-42 development-gated run.

## Reviewed invariants

- V4.1 scripts and completed artifacts remain untouched.
- Both methods start from the same base model and receive the exact same
  deterministic 108-event sequence.
- Every epoch contains 18 rescue and 18 reverse presentations.
- Both methods use the same unit-weight DPO-shift plus absolute-margin loss,
  optimizer, LoRA configuration, precision, and total budget.
- The method difference is restricted to reverse-event direction: Mask labels
  every event `B > A`; V4.2 follows frozen V4 causal direction.
- Training prompts contain only visible history, public schemas, and candidate
  actions. Official branch outcomes remain private and join after scoring.
- Offset 125 is explicitly known development data.
- Offset 165 is frozen before training, excludes four prior protocol identities,
  and is never queried unless the development gate passes.
- Raw evaluation summaries are independently recomputed by the gate.
- Snapshot, Harness, Tool-ID, continuation, V4 credit, and official evaluator
  code are not changed by V4.2.

## Validation

- Full local test suite: pass (two expected skips).
- Ruff on all new Python files: pass.
- Python compilation for all new scripts/tests: pass.
- Bash `-n` syntax check for the cloud runner: pass using Git Bash.
- `git diff --check`: pass.

## Remaining limitations

- Only three distinct reverse-credit training events exist; deterministic
  balancing necessarily repeats them six times per epoch. Confirmation on
  disjoint scenarios is therefore mandatory.
- Passing would establish controlled-state causal preference learning, not
  autonomous ToolSandbox task success.
- A failed development gate must be retained as a negative result; thresholds
  must not be tuned on offset 165.
