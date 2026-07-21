# Round 3 Review

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.7 |
| Method Specificity | 8.8 |
| Contribution Quality | 8.2 |
| Frontier Leverage | 8.0 |
| Feasibility | 7.8 |
| Validation Focus | 8.7 |
| Venue Readiness | 8.0 |

**Weighted overall: 8.54/10. Verdict: REVISE. Drift: NONE.**

## Established

The sealed-filtration estimator, fixed and stochastic potential-outcome variants, allocation-dependent variance term, local-theta timing, baseline restrictions, action-sequence definition, and LoRA score norm are now technically sound.

## Remaining blockers

1. Pilot 0 requires exact behavior-policy identity. Post-hoc log probabilities do not make old A samples on-policy. Bind the generating checkpoint/tokenizer/prompt/sampler or recollect a small clean bank.
2. The finite-batch KKT lambda requires all predicted `a,c` before sampling. Use a two-pass batch protocol: execute/seal safe B and snapshots, compute lambda across the batch, then commit probabilities and draw audits.
3. Separate theoretical sigma-field S from public model features F. Snapshots/RNG may be hashed/restored but never parsed by the residual model or allocator.
4. State expected-budget semantics, realized-cost reporting, tail probability, and feasibility `C >= p_min sum c`.
5. Add the naive B-credit bias `E[s(Q_B-Q_A)]` and the gradient mass removed by Mask.
6. Freeze task-cross-fit residuals, hidden outcomes, and propensities as mandatory Pilot-0 integrity gates.

## Decisive pilot

Use an identity-bound existing bank or recollect a small clean Full-Shadow bank. Compare uniform, residual-only, score-only, and gradient-residual allocation at 20% expected cost over at least 1,000 audit draws. Require zero-compatible bias, at least 15% gradient-MSE reduction versus uniform, improvement over residual-only, better cosine similarity, and acceptable ESS/max weight.
