# Round 2 Review

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.5 |
| Method Specificity | 8.2 |
| Contribution Quality | 7.4 |
| Frontier Leverage | 7.5 |
| Feasibility | 7.8 |
| Validation Focus | 9.0 |
| Venue Readiness | 7.4 |

**Weighted overall: 8.04/10. Verdict: REVISE. Drift: NONE.**

## Main assessment

RAPG is substantially stronger than outcome-MSE AIPW. The possible AAAI contribution is the new information structure and estimand: policy proposal A is never executed, observed return belongs to replacement B, and proposal return is revealed only through randomized Shadow. Audit allocation is an efficiency corollary, not an equal headline contribution.

## Required corrections

1. Replace the incorrect conditional claim `E[Qhat|x,A,B,G_B]=Q_A^H(x,A)` with a sealed-filtration statement. For fixed potential outcomes, `E_Z[Qhat|S,G_A]=G_A`; for stochastic Shadow, `E[Qhat|S]=E[G_A|S]`, followed by tower expectation.
2. For stochastic Shadow, the audit-dependent variance is `w^2[(1/p)E[(G_A-m)^2|S]-(E[G_A-m|S])^2]`; only the `a/p` term depends on p.
3. Audit probabilities cannot use a future PPO parameter. Lock `w=||grad log pi_theta_k(A|x)||` at collection time and state optimality only for one local gradient update at theta_k. Multi-epoch/clipped PPO is an engineering variant without this theorem.
4. Prove a plug-in bound first for the unclipped `sum a/p` objective; treat floor and ceiling with explicit boundary terms. Delete the unproved `r^2` promise if necessary.
5. Define A as a complete autoregressive tool-action sequence and define sequence log probability and the LoRA score sketch.
6. Keep `b(x)` action-independent. B and `G_B` may enter augmentation m but not the ordinary baseline because B depends on A.
7. Preserve the Harness-occupancy scope and make unassisted improvement empirical only.
8. Snapshot must precede B; restore must include RNG; the A audit must not feed back to the live B trajectory; p must be committed before any A outcome.

## Smallest decisive pilot

Freeze a small policy/LoRA, recompute proposal log probabilities and per-example LoRA score sketches for the existing Full-Shadow bank, construct the full-data gradient reference, then simulate 20% audits at least 1,000 times. Compare uniform, residual-only, score-only, and gradient-residual allocation. Continue only if RAPG bias is compatible with zero and the proposed allocator reduces held-out projected-gradient MSE by about 15% over uniform and beats residual-only without being driven by extreme propensity weights.
