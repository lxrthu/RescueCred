# AppWorld Stage 0 Code Review

**Date:** 2026-07-15  
**Status:** same-family provisional review

## Initial blocking findings

1. Official `world.evaluate()` may return a tracker requiring `to_dict()`.
2. A no-mutation `save_state/load_state` call must not be presented as proof of
   causal replay.
3. Shadow restore must include AppWorld interaction counters/log control state,
   adapter history/RNG, and process RNG, not only the database checkpoint.
4. RescueCredit's state hash must contain a deterministic DB checkpoint digest.

## Fixes applied

- Evaluation now serializes tracker-like results through `to_dict()`.
- AW0 is explicitly a contract probe; its output contains
  `authorizes_causal_shadow=false` until a real state-changing rollback test is
  implemented after inspecting the installed AppWorld contract.
- Snapshots capture and restore `environment_io`, `num_interactions`,
  `num_sub_interactions`, adapter RNG/history/steps, Python random state, and
  NumPy random state when available.
- Atomic execution no longer creates a persistent shell variable.
- Snapshot creation refuses causal replay when no deterministic checkpoint
  export digest can be found; restored DB digests must match exactly.
- Action arguments must be JSON objects, disallow NaN/Infinity, and are capped
  at 1 MB.
- AppWorld is installed in an isolated Python 3.11 environment under `/data`.
- AppWorld is pinned to `0.1.3.post1` and Freezegun to `1.5.1`, matching the
  package-declared `freezegun>=1.5.0,<=1.5.1` compatibility range. Freezegun
  1.5.5 and the incorrectly attempted 1.2.2 are both excluded.

## Local verification

- Python compilation: passed.
- Dependency-free fake-world atomic call and control-state replay: passed.
- Real AppWorld verification: pending AW0 server run.

Server compatibility note: AppWorld 0.1.3.post1 raises during `world.close()`
after an immediate `save_state(); load_state(state_id)` cycle under its own
supported Freezegun 1.5.1. AW0 therefore checks that both checkpoint methods and
a non-empty state id exist, but deliberately does not call `load_state`. The
real state-changing rollback gate remains mandatory before AW1.

## Deployment decision

Safe to deploy the CPU-only AW0 contract probe. AW1 causal auditing and all GPU
training remain blocked until a real AppWorld state-changing branch proves
`before == restored != mutated` for both DB and control state.

Second-pass reviewer verdict: no Stage 0 blockers; focused tests passed. The
probe records the AppWorld package version and describes its privacy boundary
precisely as “no ground-truth values exported.”
