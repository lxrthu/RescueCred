# Route-A Frozen Bank Code Review

Date: 2026-07-16
Review mode: local-only (team delegation was not authorized for this turn)

## Blocking checks

- PASS: public bank rejects reference actions, ground truth, and offline labels recursively.
- PASS: only AppWorld `train` is accepted by both CLI and record validator.
- PASS: public and private files are joined only by a deterministic event hash.
- PASS: Mask and RescueCredit-v2 are declared to consume the same immutable public bank hash.
- PASS: builder does not invent a causal Shadow delta or relabel exact match as causal return.
- PASS: selector prompt receives only instruction, public schema, visible receipts, proposal, and visible candidate provenance.

## Non-blocking limitations

- The bank uses controlled missing-required-argument corruptions rather than naturally sampled policy errors.
- Reference-call replay is used only to expose realistic prior receipts and compute offline diagnostics.
- This stage does not implement the AppWorld V2 training loop; it intentionally gates that work on a valid frozen bank.

## Verification

- `16 passed`: frozen-bank, AppWorld deployable Harness, and AppWorld adapter tests.
- Python bytecode compilation passed for all new modules and scripts.
- `git diff --check` produced no whitespace errors in the project files.
