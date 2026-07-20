# ToolSandbox V4.4 reference-free candidate-diversity protocol

Date: 2026-07-20

## Motivation fixed from the V4.3 data-gate result

The frozen V4.3 offset-85 multi-prefix audit produced 75 nonzero events from
33 tasks, exact snapshots, and no leakage, but only 4 reverse events from 4
tasks.  It therefore stopped before training at the preregistered 8-event,
5-task reverse gate.  Lowering that gate or training anyway is forbidden.

The failure identifies a candidate-construction bias: removing a required
argument makes corrected branch B structurally better in almost every case.
V4.4 changes only candidate construction.  It asks the deployment-visible
policy for multiple distinct schema-valid alternatives to its own proposal A,
then uses paired Shadow execution to determine direction offline.

## Frozen candidate construction

- Reuse exactly the 40 offset-85 training-only scenarios.  Offset 125 remains
  development and offset 165 remains untouched confirmation.
- Prefix actions, proposal A, and alternatives use only visible task messages,
  public Tool-ID schemas, and visible receipts.  Reference actions, milestones,
  minefields, evaluator values, and hidden state never enter generation.
- At each prefix, request up to 3 distinct alternatives to proposal A.
- A and every B must independently satisfy the public schema.  Duplicates and
  copies of A are rejected before execution.  Every candidate argument value
  must also occur in visible history, proposal A, or a public schema literal;
  unsupported invented values are discarded deterministically.
- Retain at most 4 paired events per task across H8.  A is executed only to
  advance the common reference-free prefix after all same-prefix pairs finish.
- Official ToolSandbox outcomes are used only after both candidates are fixed.

## Frozen signal/data gate

The full audit must have exact snapshots and branch prefixes, worker-failure
rate at most 0.10, at least 60 replay-valid nonzero pairs, at least 8 rescue
and 8 reverse pairs, and each direction must span at least 5 tasks.  No task
may contribute more than 4 events or 10% of the retained data.

The output is a signal/data audit only.  No adapter training is authorized
unless this gate passes.  If it passes, the next protocol may reuse the
already implemented matched anchored learner, but must freeze a new method
identity and preserve the offset-125/165 evaluation boundary.

## Interpretation boundary

Passing shows that a deployable, reference-free candidate generator can expose
bidirectional causal learning signal.  It is not yet evidence that training
improves selection or autonomous task success.
