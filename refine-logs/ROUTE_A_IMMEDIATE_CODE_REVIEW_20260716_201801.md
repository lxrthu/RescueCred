# Route-A Immediate Diagnostic Code Review

Review timestamp: 2026-07-16 20:18:01 +08:00

## Scope

Reviewed the deterministic AppWorld immediate-effect diagnostic that replays an identical reference prefix, executes A and B in separate fresh worlds, and scores each branch immediately with the official requirement evaluator.

## Integrity findings

- No Azure client, continuation worker, model generation, or GPU path is imported by the evaluator.
- Reference calls are used only to reconstruct the event-time state; actions and protected values are not written to diagnostic outputs.
- Both branches use fresh AppWorld instances with identical task, seed, call index, and prefix.
- Every invocation uses a unique output namespace, so sanity or reruns cannot reuse a stale AppWorld report.
- No reference suffix is executed after A or B.
- A failed candidate execution remains an observed branch outcome; only prefix reconstruction or missing official score invalidates the pair.
- The gate is fixed before server execution and requires signal volume, method disagreement, score improvement, win/loss improvement, and causal-direction improvement.

## Validation

- Pure unit tests cover three-way causal direction, selected-action scoring, and strict gate behavior.
- The cloud runner performs a 3-event sanity before the full pass.

## Residual limitation

This diagnostic measures immediate official-state progress at a controlled event, not end-to-end autonomous task success. A passing result supports the action-selection mechanism only.
