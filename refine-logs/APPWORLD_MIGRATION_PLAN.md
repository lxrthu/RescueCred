# RescueCredit AppWorld Migration Plan

**Date:** 2026-07-15  
**Status:** Stage AW0 implementation

API-Bank-derived controlled tasks remain a mechanism pilot. AppWorld is the
candidate paper-facing main environment, but migration is gated rather than
assumed successful.

| Stage | Scope | Hard gate |
|---|---|---|
| AW0 contract | 3 train tasks, CPU | official verify passes; public schemas and checkpoint API load; offline train scoring structure exists |
| AW1 harness audit | 30 train tasks, no policy training | eligible >= 20 across >= 10 tasks; nonzero causal >= 3; replay failures = 0; correction precision >= 0.90; harm <= 0.01 |
| AW2 one-seed pilot | Mask-v2 vs RescueCredit-v2, equal main steps | Rescue improves official dev success or first-pass and meets audit floor |
| AW3 confirmatory | frozen config, multiple seeds | official dev; test opened only once after freeze |

## Integrity boundary

- Policy actions are atomic AppWorld function calls, not arbitrary code blocks.
- `save_state/load_state` is training-time Shadow infrastructure only.
- Test-time policy execution cannot use rollback.
- Train/dev ground truth is allowed only for offline case construction, reward,
  and official evaluation.
- Ground truth, required APIs, compiled solutions, and evaluation code are
  forbidden from policy, Harness, validator, and correction-generator inputs.
- Protected AppWorld task content is never copied into this repository or its
  paste bundle.

## Compute estimate

- AW0: CPU, about 10-30 minutes including installation/download/verification.
- AW1: primarily CPU plus one inference GPU if the frozen repair generator is used.
- AW2/AW3 are not authorized until the preceding hard gates pass.
