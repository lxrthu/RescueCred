# ToolSandbox V4.3 local implementation review

Date: 2026-07-20

Verdict: **DEPLOY YES, RESULT PENDING**.

## What is fixed before the cloud run

- The measured eligible pool has 218 scenarios; indices through 204 are
  already assigned.  V4.3 never selects offset 205 or the remaining 13.
- Multi-prefix mining reuses exactly the old offset-85 training tasks and is
  capped at four uniquely identified events per task.
- Each new event stores the actual deployment-visible treatment history and
  public schemas.  Reference actions are never used, and official branch
  outcomes are removed from the training file.
- The data gate stops before training unless there are at least 60 nonzero
  events, 8 reverse events, and 5 reverse tasks, with task concentration caps.
- Mask and V4.3 consume the same deterministic 180 presentations.  Both use
  the same DPO, absolute-margin, and reference-anchor objectives; only the
  causal label routing differs.
- Offset 125 is development-only.  Offset 165 is source-bound before training
  and cannot be evaluated until the development gate passes.  The runner also
  refuses to start if prior offset-165 outcomes exist.
- The gate independently recomputes raw metrics, requires real selection
  changes and wins, and additionally requires the rescue/reverse margin-shift
  separation frozen in the plan.

## Compatibility note

`audit_toolsandbox_signal.py` and `freeze_toolsandbox_v4_protocol.py` gained a
default-one `max_events_per_scenario` field.  Default event identifiers and
single-event behavior remain regression tested, but the source hashes of these
files necessarily differ from old V4/V4.1 locks.  V4.3 therefore creates new
source-bound mining and confirmation locks; old results remain preserved and
are not re-labelled as new runs.

## Local validation

- Python compile: pass.
- Ruff on all V4.3 and modified files: pass.
- Focused ToolSandbox tests: pass.
- Full repository pytest: pass with the pre-existing skipped tests only.
- Bash syntax for `scripts/cloud/run_toolsandbox_v43_seed42.sh`: pass.
- `git diff --check`: pass.

No cloud outcome has been observed.  The correct next action is to deploy the
frozen runner once; a failed data or development gate is a valid negative
result and must not be bypassed.
