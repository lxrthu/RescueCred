# ToolSandbox V4.3 multi-prefix anchored protocol

Date: 2026-07-20

## Motivation fixed from V4.2 development evidence

V4.2 amplified reverse-margin movement by about 6.8x, but produced no sign
flip.  Reverse events moved by -0.0959 on average while rescue events also
moved toward A by about -0.0404.  V4.3 tests whether distinct visible-prefix
training states plus a conservative reference-margin anchor can turn the
directional effect into a decision change without global A drift.

The eligible single-user/multiple-tool pool contains 218 scenarios and the
existing partitions consume indices through 204, leaving only 13.  This count
was measured without executing policies or observing outcomes.  Therefore the
protocol expands states inside the already designated offset-85 training
tasks; it does not consume development or confirmation tasks.

## Frozen training-data construction

- Reuse the exact 40 offset-85 training scenarios.
- Follow only reference-free worker proposals and visible receipts.
- Continue the common visible prefix after a treatment instead of stopping at
  the first eligible point.
- Retain at most four distinct treatment events per scenario across H8.
- Every retained row binds its actual visible history and public tool schemas.
- Official outcomes remain private causal labels and never enter prompts.

The data gate requires at least 60 nonzero events, 8 reverse events from at
least 5 tasks, no more than 4 events per task, and maximum task share 0.10.
Failure stops before training.  The original aspirational target of 20 reverse
events is infeasible from the 13 unused scenarios alone; the frozen lower gate
tests diversity rather than repeated copies of the original three examples.

## Frozen learner comparison

- Methods: matched Mask and V4.3.
- Same deterministic 60-event sequence per epoch, 30 rescue and 30 reverse.
- Three epochs, 180 total presentations, same model/optimizer/LoRA/precision.
- Both use unit-weight DPO shift plus absolute margin (`coef=1.0`, target 0.05).
- Both add the same reference-margin anchor
  `0.25 * (policy_margin - base_margin)^2`.
- Mask labels every event B>A; V4.3 follows frozen causal direction.

## Evaluation boundary and gates

- Offset 125 remains known development data.
- Offset 165 remains untouched confirmation data.  A new source-bound lock must
  reproduce the exact scenario hashes frozen before V4.2 and must verify that
  the V4.2 confirmation audit was never created.
- Development must have at least one selection disagreement, strictly positive
  causal-accuracy improvement, more wins than losses, terminal/progress
  noninferiority, and reverse margin shift at least 0.02 more negative than the
  rescue margin shift.
- Confirmation retains the stricter V4.2 thresholds: at least three
  disagreements, at least 0.05 causal-accuracy gain, wins over losses, and
  terminal/progress noninferiority, plus the same class-conditional shift gate.

Passing remains controlled-state causal preference evidence, not autonomous
ToolSandbox task-success evidence.
