# RAPG frozen task-stability audit

## Status and boundary

Pilot 0 failed its preregistered surrogate gate. This audit does not reopen
model selection, change thresholds, or convert that failure into a pass. It
uses only the frozen bank, cross-fit predictions, propensities, and gate result.

## Fixed analyses

1. Recompute Uniform and RAPG design-variance contributions per task.
2. Run a 20,000-replicate task-cluster bootstrap of the aggregate MSE gain.
3. Compute leave-one-task-out gains and top-task contribution concentration.
4. Recompute expected audit costs with `math.fsum` and classify the prior
   matched-cost miss as numerical only when every absolute error is at most
   `1e-5`.

## Decision

- `surrogate_signal_robust_but_frozen_gate_failed` requires both a positive
  bootstrap lower bound and at least 15% gain in every leave-one-task-out run.
  It authorizes only a fresh preregistered on-policy pilot.
- Otherwise the result is `surrogate_gain_task_concentrated`, and RAPG scaling
  stops.

In either case, the current paper-facing positive claim remains unsupported.
