# Round 3 Refinement

## Problem Anchor

- **Bottom-line problem:** A runtime Harness executes correction B instead of policy proposal A, so the assisted return identifies the executed system but not the credit attributable to the trainable proposal. We need a method that improves the unassisted policy under a bounded audit budget while deployment continues to default safely to B.
- **Must-solve bottleneck:** Rescue/Reverse direction is not reliably identifiable from deployment-visible static context, one-step receipts, two-step receipts, fixed state observers, or simple goal contracts. A method cannot stably learn an event-level label from an observation interface that erases the relevant counterfactual information.
- **Non-goals:** Do not tune another preference/RL loss on the same labels; do not require a perfect per-event A/B router; do not expose reference actions, official evaluators, or hidden database state to the deployed policy; do not claim a universally reliable automatic Harness.
- **Constraints:** Preserve safe default execution B; use exact isolated Shadow only on a budgeted subset; audit probabilities and predictors must be fixed before each draw; current banks are development-only and fresh task-disjoint confirmation is required; compute should fit the existing ToolSandbox/API-Bank pipeline and at most a small model pilot.
- **Success condition:** At the same Shadow interaction cost, the method has lower counterfactual-credit MSE than uniform auditing, remains unbiased under adaptive predictable sampling, and produces a measurable improvement in unassisted/first-pass policy behavior over Mask + Correction without reducing assisted safety.

## Anchor Check

No change of problem or success condition. This round closes experimental identity, online-allocation, and public-feature loopholes.

## Simplicity Check

RAPG remains the sole headline. The method uses one two-headed public-feature model and a two-pass batch audit. No router, contract, or extra verifier is added.

## Final Candidate: RAPG / AdaAudit

### A. Action-substitution bias is the problem object

For the first path-changing intervention at frozen behavior parameters `theta_k`, proposal A is the full autoregressive tool-call sequence sampled by `pi_theta_k`, while the Harness deterministically or stochastically maps `(x,A)` to executed B. Let `s_k=grad log pi_theta_k(A|x)` and let `Y_A,Y_B` denote bounded returns of the two potential branches under a fixed continuation.

Crediting the observed B return to A estimates the wrong coupling:

```text
g_Bcredit - g_proposal = E[s_k (Y_B-Y_A)].
```

Masking every intervention instead deletes

```text
g_masked_missing = E[1{H replaces A} s_k Y_A],
```

the precise gradient mass associated with corrected proposals. These decompositions explain why “always trust” and masking can preserve assisted success yet fail to improve the underlying proposal policy.

### B. Estimand and information boundary

The theoretical target is the single local proposal gradient

```text
g_k = E_{x~d^{H,pi_k}, A~pi_k}[s_k(x,A) Y_A].
```

Only the first path-changing intervention is included and occupancy is fixed at `d^{H,pi_k}`. Unassisted end-to-end improvement is empirical only.

Two objects are now separated:

- `S_t`: the mathematical pre-audit sigma-field. It includes snapshot/RNG commitments required to define the potential outcome, plus sealed public information.
- `F_t = phi(x,A,B,G_B,public ledger metadata)`: the only feature vector visible to the augmentation model and allocator.

Exact snapshot/database/RNG contents may only be hash-bound, restored, and checked for integrity. They may never be parsed into F or provided to a learned component.

### C. RAPG theorem

Before the audit draw, `m_t=m_psi(F_t)` and `p_t>0` are committed. With fixed event-level potential outcome Y_A and `Z|S ~ Bernoulli(p)`, define

```text
Yhat_A = m(F) + Z/p * (Y_A-m(F)),
ghat = s_k * stopgrad(Yhat_A-b(x)),
```

where b is action-independent. Then

```text
E_Z[Yhat_A | S,Y_A] = Y_A,
E[ghat] = g_k.
```

If Shadow contains irreducible randomness, `E[Yhat_A|S]=E[Y_A|S]` and tower expectation gives the conditional-mean version. B and `G_B` may enter m(F), but not b(x). All estimator/allocator paths are stop-gradient. The theorem covers one on-policy local update at theta_k; multi-epoch/clipped PPO has no unbiasedness or allocation-optimality claim.

### D. Gradient-aware expected-budget allocation

At collection time compute and seal the whole-sequence LoRA score vector `s_k` and `w=||s_k||`. Let

```text
a_t = w_t^2 E[(Y_A-m(F_t))^2 | S_t].
V(p)=sum_t a_t/p_t.
```

For expected Shadow cost `sum p_t c_t <= C`, the oracle interior KKT solution is `p*=lambda sqrt(a/c)`, clipped to `[p_min,1]`. Feasibility requires `C >= p_min sum c`. The result is only an efficiency corollary for the local RAPG estimate.

Main results report `E[sum Zc]`, realized cost, 95th/99th cost quantiles across draws, ESS, maximum inverse propensity, and any ceiling/floor saturation. Reaching a realized cap never forces future p to zero. A hard-cap design is outside the main theorem.

### E. Implementable two-pass batch protocol

For a finite batch of N first-intervention events:

1. Freeze and hash `theta_k`, tokenizer, prompt template, decoding configuration, Harness, continuation, and scorer.
2. For every event, sample A with this exact policy, save a pre-B snapshot/RNG commitment, execute safe B, seal `G_B`, compute public F, score `s_k`, cost estimate c, and cross-fitted `m,ahat`.
3. After all `ahat,c` are sealed, solve one batch dual lambda using only predictions and the expected budget C.
4. Ledger-bind every p before opening any A outcome, then independently draw all Z values.
5. Execute isolated A Shadow only for selected snapshots; construct RAPG; take one local update.
6. Only after the batch closes, update `m,ahat` for a future batch.

This protocol exactly implements the finite-batch KKT allocation and requires snapshot persistence only until the audit draw.

### F. Behavior-policy identity gate

Pilot 0 may use an existing bank only if its manifest proves byte-level/hash identity for:

- proposal generator checkpoint and adapter;
- tokenizer and prompt serialization;
- decoding temperature/top-p and sampling implementation;
- tool schema ordering;
- per-event RNG seed;
- stored proposal A.

The audit script regenerates A and requires exact equality. A post-hoc log probability is insufficient. If any field is missing or any replay differs, Pilot 0 recollects a small clean bank with the frozen LoRA policy and Full Shadow for every event. Off-policy repair is not the default because sequence importance ratios would dominate variance.

### G. Residual-model integrity gate

For every held-out task fold:

1. train `m,ahat` only on other task groups;
2. freeze their hashes before processing held-out F;
3. keep held-out Y_A sealed until all p values and the batch lambda are committed;
4. compute lambda from `ahat,c` only, never from true residuals;
5. in Monte Carlo simulation, keep F, predictions, and p fixed and redraw only Z;
6. recompute all metrics from bound prediction ledgers.

Oracle allocation using true residuals is reported only as an unattainable upper bound, never as AdaAudit.

### H. Pilot 0: smallest decisive result

Use an identity-valid existing bank or recollect roughly 100--150 clean Full-Shadow first-intervention events from task-disjoint families. Freeze one small LoRA policy. Construct the full-data adapter-gradient reference from all Y_A.

At 20% expected cost and at least 1,000 independent audit draws compare:

- uniform Bernoulli;
- residual-only `sigma/sqrt(c)`;
- score-only `w/sqrt(c)`;
- AdaAudit `w*sigma/sqrt(c)`;
- oracle true-residual allocation as ceiling.

Pre-registered go gate:

- RAPG projected-gradient bias confidence interval includes zero and coverage is calibrated;
- held-out gradient MSE is at least 15% below uniform;
- AdaAudit beats residual-only, proving score weighting contributes;
- gradient cosine similarity improves;
- gain persists after capping/reporting extreme weights and is not concentrated in one task family;
- expected-cost feasibility and realized-cost tails pass the frozen budget report.

If the bank identity gate fails, recollection is mandatory. If the MSE gate fails, stop AdaAudit before policy training.

### I. Policy-level claim gate

Only after Pilot 0 passes, compare Mask + Correction, uniform RAPG, AdaAudit RAPG, and Full Shadow using the same sequence of local updates and matched Shadow cost. Fresh-task unassisted first-pass success is primary; assisted safety is a non-inferiority constraint. Without a measurable unassisted gain over both Mask + Correction and uniform RAPG, the work may claim an estimator result but not autonomous policy improvement.

### J. Contribution boundary

1. Primary: formalize runtime action substitution as missing proposal-return gradient, including the explicit bias of naive B-credit and the gradient mass lost by masking.
2. Method: RAPG restores an unbiased local proposal gradient through propensity-bound randomized Shadow; gradient-aware two-pass auditing is the cost mechanism.
3. Evidence: matched-budget gradient reconstruction followed, only if successful, by policy-level unassisted evaluation.

The square-root allocation, HT augmentation, and importance sampling are not independently claimed as novel. Novelty rests on the proposal/execution information structure, estimand, and demonstrated policy consequence.
