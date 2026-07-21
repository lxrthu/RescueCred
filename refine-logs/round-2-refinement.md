# Round 2 Refinement

## Problem Anchor

- **Bottom-line problem:** A runtime Harness executes correction B instead of policy proposal A, so the assisted return identifies the executed system but not the credit attributable to the trainable proposal. We need a method that improves the unassisted policy under a bounded audit budget while deployment continues to default safely to B.
- **Must-solve bottleneck:** Rescue/Reverse direction is not reliably identifiable from deployment-visible static context, one-step receipts, two-step receipts, fixed state observers, or simple goal contracts. A method cannot stably learn an event-level label from an observation interface that erases the relevant counterfactual information.
- **Non-goals:** Do not tune another preference/RL loss on the same labels; do not require a perfect per-event A/B router; do not expose reference actions, official evaluators, or hidden database state to the deployed policy; do not claim a universally reliable automatic Harness.
- **Constraints:** Preserve safe default execution B; use exact isolated Shadow only on a budgeted subset; audit probabilities and predictors must be fixed before each draw; current banks are development-only and fresh task-disjoint confirmation is required; compute should fit the existing ToolSandbox/API-Bank pipeline and at most a small model pilot.
- **Success condition:** At the same Shadow interaction cost, the method has lower counterfactual-credit MSE than uniform auditing, remains unbiased under adaptive predictable sampling, and produces a measurable improvement in unassisted/first-pass policy behavior over Mask + Correction without reducing assisted safety.

## Anchor Check

The revision remains about proposal credit under action replacement. It makes the probability-timing and state-distribution scope narrower and correct; it does not introduce another router or hidden signal.

## Simplicity Check

RAPG is the only headline algorithm. Gradient-aware auditing is its budget-efficiency mechanism. The prior non-identifiability and failed-router results motivate why an intervention channel is required.

## Revised Proposal: RAPG with Predictable Gradient-Aware Auditing

### 1. Action and local objective

At optimization iteration k, freeze behavior parameters `theta_k`. A proposal `A=(a_1,...,a_L)` is the entire autoregressive token sequence encoding the first path-changing tool call:

```text
log pi_theta(A|x) = sum_j log pi_theta(a_j | x,a_<j).
s_k(x,A) = grad_theta log pi_theta(A|x) evaluated at theta_k.
```

The exact theoretical target is the current local gradient

```text
g_k = E_{x~d^{H,pi_k}, A~pi_k}[s_k(x,A) Q_A^H(x,A)].
```

Only the first path-changing intervention is included. `d^{H,pi_k}` is held fixed during the local update. This is not claimed to be the gradient of fully unassisted end-to-end return.

### 2. Sealed filtration and potential outcome

Let `S_t` be the audit-time sigma-field containing agent-visible history, pre-B exact snapshot and committed RNG state, A, B, the sealed live B trajectory and `G_B`, model versions, and every quantity used to choose `p_t`, but no result from executing A in Shadow.

The main theorem uses a **fixed event-level potential outcome** `Y_t^A`: restore the committed snapshot/RNG, execute A, and apply the frozen continuation and bounded scorer. Thus `Y_t^A` is fixed before the Bernoulli audit draw, even though it is unobserved when `Z=0`. Define

```text
m_t = m_psi(S_t)
Yhat_t^A = m_t + Z_t/p_t * (Y_t^A-m_t).
```

where `m_t` and `p_t>0` are `S_t`-measurable and stop-gradient, and `Z_t|S_t ~ Bernoulli(p_t)`.

**Theorem 1 (audit-unbiased potential outcome).**

```text
E_Z[Yhat_t^A | S_t,Y_t^A] = Y_t^A.
```

Therefore

```text
E[s_k Yhat^A] = E[s_k Y^A] = g_k.
```

For environments where restore cannot fix all randomness, redefine `Y^A` as a fresh frozen-continuation draw. Then `E[Yhat^A|S]=E[Y^A|S]`, and tower expectation yields the gradient of the conditional-mean target. The proof never equates `E[Y^A|S]` with `E[Y^A|x,A]` without tower expectation.

The ordinary baseline is restricted to `b(x)` and remains action-independent. B and `G_B`, which depend on A, enter only the augmentation model `m(S)`.

### 3. RAPG estimator

For the theorem-backed on-policy local update:

```text
ghat_t = s_k(x_t,A_t) * stopgrad(Yhat_t^A - b(x_t)).
```

Every occurrence of `m`, `p`, `Z`, `Y`, and the score-norm allocator is detached. A single local REINFORCE/natural-gradient/very-small-step update is theorem-backed. Multiple PPO epochs, likelihood-ratio clipping, or recomputing the score at future theta are implementation variants and receive no unbiasedness or allocation-optimality claim.

### 4. Correct gradient-variance objective

Let `w_t=||s_k(x_t,A_t)||_2`, computed and sealed at collection time, `r_t=Y_t^A-m_t`, and `a_t=w_t^2 E[r_t^2|S_t]`. For stochastic Shadow,

```text
Var(ghat_t | S_t)
 = w_t^2[(1/p_t)E[r_t^2|S_t] - (E[r_t|S_t])^2]
```

up to audit-independent baseline/mean terms and the direction of the score vector. Therefore the only allocation-dependent trace term is

```text
V(p) = sum_t a_t/p_t.
```

For fixed potential outcomes this reduces to the same p-dependent term; subtracting `sum a_t` gives the familiar `(1/p-1)` audit-noise expression, but optimization is stated using `V(p)`.

### 5. Oracle allocation as an efficiency corollary

Under an expected cost budget `sum p_t c_t <= C`, `0<p_t<=1`, the KKT solution for non-saturated events is

```text
p_t* = lambda sqrt(a_t/c_t) = lambda w_t sigma_t/sqrt(c_t),
```

with clipping at 1 and a pre-registered positive `p_min` when required for weight control. This result is an efficiency corollary of RAPG, not a separate novelty claim.

The primary method learns one shared augmentation model with two heads: `m(S)` and `ahat(S)`, the latter predicting the squared gradient-weighted residual. For a LoRA policy, `w_t` is the norm of the per-example LoRA gradient of the whole action-sequence log probability; a fixed diagonal-Fisher/random-projection sketch is the cheap ablation, and exact LoRA norms are used in the smallest pilot when feasible.

### 6. Plug-in result with honest scope

The paper will prove the following only for the unclipped interior problem. If `a_t/r <= ahat_t <= r a_t` and oracle/plugin allocations satisfy the same expected cost exactly, then substituting their closed forms into `V(p)=sum a/p` gives a finite multiplicative excess bound. The exact constant will be derived in the appendix; no `r^2` claim is made before proof verification.

For `p_min` and `p=1`, the statement becomes the interior bound plus an explicit additive boundary term over events whose active sets differ. If this proof is not completed, the learned allocator is evaluated empirically and only oracle optimality is claimed.

### 7. Snapshot and causality protocol

1. Before executing B, save the exact environment snapshot and RNG commitment.
2. Execute B only on the live branch; seal its public trajectory and `G_B`.
3. Compute and irreversibly ledger-bind `m`, `ahat`, `w`, `p`, feature hashes, and allocator version.
4. Draw Z. Only Z=1 opens an isolated A branch restored from the pre-B snapshot/RNG.
5. Never feed A-branch observations back into the live B trajectory.
6. Update residual parameters only after the event ledger is sealed.

### 8. Algorithm

```text
freeze theta_k, Harness, continuation, scorer
for each trajectory:
    reach first path-changing intervention x
    sample full proposal sequence A~pi_theta_k; compute logp and LoRA score s_k
    hash A; save exact pre-B snapshot/RNG
    generate/hash B; execute B safely; seal G_B
    predict m and ahat from S; allocate and commit p using sealed w=||s_k||
    draw Z~Bernoulli(p)
    if Z=1: restore isolated snapshot/RNG; execute A -> Y_A
    Yhat <- m + Z/p*(Y_A-m)
    store s_k*stopgrad(Yhat-b(x))
average RAPG samples; take one local policy update
after sealing, update m/ahat from audited residuals for the next batch
```

### 9. Novelty statement

Active importance sampling selects among rewards from actions that were actually executed. RAPG addresses a different coupling: policy action A was sampled but never executed; observed `G_B` belongs to a semantically different replacement `B=H(x,A)`; assigning `G_B` to `log pi(A|x)` is invalid; randomized exact Shadow creates the otherwise missing A-return channel with recorded propensity. The contribution is the formal action-substitution estimand and unbiased proposal-gradient estimator. Allocation merely makes that estimator affordable.

### 10. Decisive experiment ladder

**Pilot 0 -- gradient reconstruction, before training.** Freeze one LoRA policy. On the existing Full-Shadow development bank, recompute complete-action log probabilities and exact/sketched LoRA score vectors. Treat full `Y_A` as the reference gradient. In task-held-out streams, simulate 20% audit at least 1,000 times for uniform, residual-only, score-only, and `w*sigma/sqrt(c)` allocation.

Go only if:

- projected/full adapter-gradient bias is statistically compatible with zero;
- proposed allocation lowers held-out gradient MSE at least 15% versus uniform;
- it beats residual-only, showing the score norm matters;
- cosine similarity improves and no result is driven by tiny-p weights.

**Pilot 1 -- policy effect.** If Pilot 0 passes, compare Mask + Correction, uniform RAPG, AdaAudit RAPG, and Full Shadow in a small fixed-compute local-update experiment. Primary outcome is fresh-task unassisted first-pass success; assisted safety is a non-inferiority constraint.

**Confirmation.** Freeze the method and use tasks disjoint from every existing development bank and model-selection decision. If unassisted policy gain fails, the paper may report an estimator result but must not claim autonomous improvement.

### 11. Compressed contribution list

1. Formalize runtime action substitution as a missing proposal-return gradient problem and state its observational non-identifiability boundary.
2. Derive RAPG, an audit-unbiased gradient estimator; predictable gradient-aware Shadow allocation is its cost-control mechanism.
3. Validate at the policy level under matched audit cost, with an explicit stop rule if lower gradient error does not translate to unassisted behavior.
