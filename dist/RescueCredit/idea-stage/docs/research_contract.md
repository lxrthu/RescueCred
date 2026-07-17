# Research Contract: RescueCredit / Compensation Trap

## Selected Idea

- **Description**: Runtime corrective harnesses can map bad and good policy actions to the same successful execution, hiding the original action's potential return. RescueCredit restores this missing credit using deterministic local labels or randomized shadow residual estimates, then routes credit by token provenance and adds verified correction preference.
- **Source**: `RescueCredit_Technical_Roadmap_for_Codex_CN.md`, V3.
- **Selection rationale**: The claim is falsifiable in an exact MDP and a controlled tool benchmark, with a clear negative-result rule against Mask + Correction.

## Core Claims

1. Strong many-to-one correction can raise assisted success while reducing credit fidelity for the policy proposal.
2. A commit-before-draw randomized residual estimator is unbiased when its probability and control variate are fixed before audit.
3. RescueCredit is useful only if it improves unassisted/first-pass behavior or matches Full Shadow at meaningfully lower interaction cost.

## Method Summary

The implementation logs the first teachable intervention, snapshots pre-intervention state, obtains `G0` from a narrowly scoped deterministic verifier or from randomized shadow execution, and updates Patch EMA only after the current estimate. Policy prefix tokens receive estimated original-action credit, policy suffix tokens receive assisted return, and harness/tool/environment tokens have zero policy gradient. Verified corrected actions supply a pairwise preference signal.

## Experiment Design

- **Datasets**: exact Rescue-MDP; API-Bank-derived controlled train/dev/test_id/test_tool_ood.
- **Baselines**: Naive H+GRPO, Mask + Correction, Full Shadow.
- **Metrics**: S_on, S_off, First-pass, IR, DG, CF, G0 bias/MSE, interaction cost.
- **Compute**: one-seed gate, then at most three seeds on 4×H200.

## Status

- [x] Idea selected
- [x] Core method implemented
- [x] Deterministic infrastructure tests
- [ ] H200 pilot results
- [ ] Confirmatory results
- [ ] Ablations

