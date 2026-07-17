# Route-A Bounded-Horizon Code Review

Review route: fresh Codex reviewer, same-family provisional assurance.

## Initial verdict: DEPLOY NO

The first review found that the protocol could be relabeled with arbitrary horizons/seeds, worker failures could be mistaken for policy stops, H4/H8 prefix coupling was not observed directly, cache identity was incomplete, replay seeds could shift when event construction skipped tasks, leakage checks were self-attested, score fallback was heuristic, and invalid rows could satisfy disagreement gates.

## First remediation and second verdict: DEPLOY NO

The implementation added frozen protocol constants, worker error status, policy fingerprints, exact trace-prefix checks, task-index replay seeds, strict report parsing, and valid-pair disagreement counting. The second review then found four remaining blockers: overrideable gate thresholds, incomplete prefix/initial-action error detection, weak sanity/exit semantics, and incomplete checkpoint fingerprints.

## Final remediation

- Gate thresholds are fixed at 40 valid and 5 nonzero events; CLI overrides were removed.
- Gate independently verifies the exact event hash, event-set hash, 55-event count, seed 42, horizons 4/8, embedded pre-outcome lock, and Mask/V2 file hashes.
- Prefix execution uses robust failure markers; initial action Python exceptions invalidate the branch.
- Sanity requires 3/3 valid H8 pairs, verified H4/H8 prefixes, and a conflict-free cache.
- Full gate failure exits nonzero.
- Checkpoint identity now includes all behavioral helper hashes, Python/AppWorld/freezegun versions, dev split hash, all task instructions/reference-call fixtures, and per-task public API docs.
- Checkpoint filename and embedded event IDs must agree.
- Worker timeout aborts rather than desynchronizing the JSONL protocol.

## Final verdict: DEPLOY YES

No blocking issue remained. Ten focused tests and Python compilation passed. Live AppWorld/Azure execution remains covered by the mandatory three-event remote sanity.

- review_independence: same-family
- acceptance_status: provisional
