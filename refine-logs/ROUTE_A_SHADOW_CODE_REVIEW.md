# Route-A Shadow Credit Code Review

Date: 2026-07-16
Review mode: local-only (team delegation was not authorized for this turn)

## Blocking checks

- PASS: A and B branches use separate AppWorld instances with identical seed and identical reconstructed event prefix.
- PASS: continuation policy receives no reference action, protected evaluator value, private audit label, or reference suffix.
- PASS: only the official AppWorld evaluator is accepted as the branch return; missing metrics invalidate replay instead of falling back to a model judgment.
- PASS: output credit is keyed to the immutable public-bank event ID and records the bank SHA-256.
- PASS: private exact-match audit file is never opened.
- PASS: smoke gate blocks training unless at least 10 branch pairs are valid and at least 3 have nonzero causal support.
- PASS: Azure key remains environment-only and is not embedded in code or paste bundle.

## Explicit scope limitation

The train reference prefix is used only to reconstruct a controlled AppWorld event state. It is never shown to the continuation model or emitted to a training artifact. Consequently this is a controlled-state causal-credit benchmark, not an end-to-end reference-free agent rollout.

## Verification

- `21 passed`: Shadow credit, frozen bank, AppWorld adapter, and deployable Harness tests.
- Bytecode compilation passed for every new Python entrypoint.
- Secret scan found no embedded API key or key assignment.
