# ToolSandbox V4.2 balanced-margin protocol (frozen design)

Date: 2026-07-20

## Question

V4.1 moved all three reverse-credit margins in the correct direction, but the
movement was too small to cross zero.  V4.2 tests whether this is a learner
strength problem rather than a Harness or causal-credit failure.

## Locked change

Only the preference learner changes:

- use the same 36 frozen offset-85 training events as V4.1;
- present exactly 36 events per epoch for three epochs to both methods;
- deterministically balance each epoch to 18 rescue and 18 reverse events;
- give Mask and V4.2 the identical event sequence;
- train both methods with the identical DPO-shift plus absolute-margin loss;
- set `absolute_margin_coef=1.0` and `target_margin=0.05` for both methods;
- Mask keeps the non-causal `B > A` label on every presentation;
- V4.2 uses the frozen V4 causal direction (`B > A` for rescue and `A > B`
  for reverse).

No Harness prompt, Tool-ID interface, continuation policy, V4 lexicographic
credit, official evaluator, snapshot logic, model, LoRA rank, optimizer,
learning rate, epoch count, or total presentation budget changes.

## Data roles

- Training: V4.1 offset-85 audit, already frozen and passed.
- Development: V4.1 offset-125 evaluation.  Its outcomes are already known and
  it is explicitly a development diagnostic, never confirmatory evidence.
- Confirmation: offset 165, exactly 40 scenarios, H8/search8, frozen before
  either V4.2 adapter is trained.  It must be disjoint from the V4 offset-40,
  V4.1 diagnostic offset-80, V4.1 training offset-85, and V4.1 development
  offset-125 protocol locks.

Official branch outcomes are never placed in model prompts or training files.
They are joined only after candidate scoring.

## Development gate

Before any offset-165 API calls, V4.2 must satisfy all integrity checks and:

- at least 20 valid nonzero events and at least 2 reverse events;
- at least one Mask/V4.2 selection disagreement;
- strictly positive causal-accuracy improvement;
- more V4.2 wins than losses on disagreements;
- no terminal-similarity or progress-AUC regression beyond `1e-12`.

Failure stops the run before confirmation.

## Confirmation gate

The untouched offset-165 set requires:

- at least 20 valid nonzero events and at least 2 reverse events;
- at least 3 selection disagreements;
- causal-accuracy improvement of at least 0.05;
- more V4.2 wins than losses;
- terminal-similarity and progress-AUC noninferiority.

This remains a controlled-state preference diagnostic, not autonomous task
success evidence.
