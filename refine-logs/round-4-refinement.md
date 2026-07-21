# Round 4 Refinement

## Problem Anchor

- **Bottom-line problem:** A runtime Harness executes correction B instead of policy proposal A, so the assisted return identifies the executed system but not the credit attributable to the trainable proposal. We need a method that improves the unassisted policy under a bounded audit budget while deployment continues to default safely to B.
- **Must-solve bottleneck:** Rescue/Reverse direction is not reliably identifiable from deployment-visible static context, one-step receipts, two-step receipts, fixed state observers, or simple goal contracts. A method cannot stably learn an event-level label from an observation interface that erases the relevant counterfactual information.
- **Non-goals:** Do not tune another preference/RL loss on the same labels; do not require a perfect per-event A/B router; do not expose reference actions, official evaluators, or hidden database state to the deployed policy; do not claim a universally reliable automatic Harness.
- **Constraints:** Preserve safe default execution B; use exact isolated Shadow only on a budgeted subset; audit probabilities and predictors must be fixed before each draw; current banks are development-only and fresh task-disjoint confirmation is required; compute should fit the existing ToolSandbox/API-Bank pipeline and at most a small model pilot.
- **Success condition:** At the same Shadow interaction cost, the method has lower counterfactual-credit MSE than uniform auditing, remains unbiased under adaptive predictable sampling, and produces a measurable improvement in unassisted/first-pass policy behavior over Mask + Correction without reducing assisted safety.

## Anchor Check

The final revision changes no component. It makes the action-substitution theorem package and falsification thresholds explicit.

## Simplicity Check

One problem object, one estimator, one budget mechanism, one pre-training pilot. Allocation and augmentation are not separately marketed as novel.

# Final Candidate: RAPG -- Randomized Proposal Credit behind Runtime Replacement

## Core statement

When a Harness replaces proposal A with executed action B, the log-probability belongs to A but the observed return belongs to B. RAPG preserves safe execution of B while randomly revealing a bounded subset of missing A returns through exact isolated Shadow, then uses recorded propensities to recover the local proposal-policy gradient.

## Formal action-substitution package

At frozen iteration `theta_k`, A is the full autoregressive tool-call sequence sampled from `pi_k`, `s_k=grad log pi_k(A|x)`, B is the Harness replacement, and `Y_A,Y_B` are bounded potential returns under the same committed continuation. The target is

```text
g_k = E_{x~d^{H,pi_k},A~pi_k}[s_k Y_A].
```

The scope is the first path-changing intervention and fixed Harness occupancy. It is not the full unassisted-return gradient.

### Theorem 1: Replacement bias and masking deletion

Naively assigning the executed B return to proposal A estimates

```text
g_Bcredit = g_k + E[s_k(Y_B-Y_A)].
```

Masking all replaced events deletes

```text
Delta_mask = E[1{A replaced} s_k Y_A]
```

from the proposal gradient. These terms are zero only under special assumptions not guaranteed by a runtime Harness.

### Theorem 2: No-audit non-identification

Let the ordinary repaired log be `O=(x,A,B,Y_B,F)`. If two latent environments induce the same distribution of O but different conditional `Y_A` and hence different `g_k`, no estimator measurable with respect to O identifies `g_k` in both. Static classifiers, receipts, goal contracts, and B-return augmentation remain within this boundary unless they add an observation separating the two worlds.

### Theorem 3: RAPG identification

Let S be the mathematical pre-audit sigma-field and `F=phi(x,A,B,Y_B,public ledger metadata)` the only learned-model features. Snapshot/RNG contents may be hashed and restored but never parsed into F. Commit `m(F)` and `p>0`, draw `Z|S~Bernoulli(p)`, and define

```text
Yhat_A = m(F) + Z/p (Y_A-m(F)),
ghat = s_k * stopgrad(Yhat_A-b(x)).
```

For a fixed event-level potential outcome,

```text
E_Z[Yhat_A|S,Y_A]=Y_A,   E[ghat]=g_k.
```

For stochastic Shadow, `E[Yhat_A|S]=E[Y_A|S]` and tower expectation identifies the conditional-mean target. `b(x)` is action-independent; B and `Y_B` enter only m. All estimator and allocator paths are detached. The claim covers one local on-policy update at theta_k, not clipped or multi-epoch PPO.

### Corollary: Budget efficiency

With `w=||s_k||`, `a=w^2 E[(Y_A-m(F))^2|S]`, fixed known pilot cost `c=1`, and expected audit budget `sum p<=C`, the allocation-dependent gradient second moment is `V(p)=sum a/p`. The finite-batch interior optimum is `p*=lambda sqrt(a)`, clipped to `[p_min,1]`. A cross-fitted public-feature model predicts a; this plug-in allocator is evaluated empirically. No unproved global plug-in bound is required for the main contribution.

## Implementable protocol

1. Freeze and hash policy/adapter, tokenizer, prompt serialization, sampler, Harness, continuation, scorer, fixed Shadow horizon, and `p_min`.
2. First pass: for each event sample A from that exact policy; save pre-B snapshot/RNG commitment; execute safe B; seal Y_B; compute public F, full-sequence LoRA score s, and task-cross-fitted `m,ahat`.
3. After the full batch is sealed, solve lambda from `ahat` only for a 20% expected audit rate.
4. Ledger-bind every p before opening any A outcome, then independently draw Z.
5. Run exact isolated A Shadow only where Z=1; never feed it back to the live B trajectory.
6. Form RAPG and take one local update. Update residual models only for later batches.

Expected budget means `E[sum Z]<=C`; report realized mean, 95th/99th quantiles, ESS, and max `1/p`. Feasibility requires `C>=Np_min`. The primary estimator always uses the true committed `1/p` without clipping. Weight clipping is explicitly labeled a biased diagnostic ablation.

## Pilot 0 integrity gates

Existing data are usable only if A regenerates exactly under hash-identical checkpoint, adapter, tokenizer, prompt, sampler, tool schema, and seed. Otherwise recollect 100--150 clean first-intervention Full-Shadow events using the frozen LoRA policy. Post-hoc likelihood scoring is not accepted as behavior identity.

For every task-held-out fold, freeze m/ahat before held-out processing, seal all Y_A until every p and lambda are committed, compute lambda from predictions only, and then repeat 1,000 audit draws with fixed propensities. Oracle true-residual sampling is an unattainable ceiling only.

## Pilot 0 comparisons and gates

At fixed `c=1`, `p_min`, and 20% expected audit rate compare uniform, residual-only, score-only, AdaAudit/RAPG `w*sigma`, and oracle residual ceiling.

Primary results use unmodified inverse propensities. Pre-register:

- **bias equivalence:** `||E[ghat]-g_full|| <= 0.05 ||g_full||`, tested by a two-one-sided equivalence procedure on fixed random gradient projections with multiplicity correction;
- **efficiency:** held-out gradient MSE at least 15% below uniform;
- **mechanism:** significantly lower MSE than residual-only;
- **quality:** improved cosine similarity to full gradient;
- **robustness:** gain is not concentrated in one task family and passes ESS/max-weight diagnostics.

Failure of behavior identity forces recollection. Failure of bias equivalence or efficiency stops the method before policy training.

## Policy gate

Only after Pilot 0 passes, compare Mask + Correction, uniform RAPG, AdaAudit RAPG, and Full Shadow under matched cost and the same local-update schedule. Primary outcome is fresh-task unassisted first-pass success; assisted safety is a non-inferiority constraint. Without gain over both Mask + Correction and uniform RAPG, do not claim autonomous improvement.

## Related-work discriminant

| Setting | Was policy action A executed? | Whose reward is observed? | Missing channel | RAPG difference |
|---|---:|---|---|---|
| Active importance-sampled policy gradient | Yes | A | sample selection only | no action-return mismatch |
| Shield/intervention learning | Often no | B or safety label | usually learns intervention/expert target | does not recover A potential return |
| Verifier-based agent RL | Yes | executed trajectory | verifier score | verifier scores actual behavior |
| RAPG | No | replacement B | proposal return Y_A | randomized Shadow identifies proposal gradient |

## Contribution claim

The main contribution is the three-step action-substitution result: runtime replacement creates explicit B-credit bias and masking deletion; proposal gradient is not identified by ordinary repaired logs; predictable randomized Shadow plus RAPG identifies it without sacrificing safe B execution. Gradient-aware allocation is the practical budget mechanism. Policy-level improvement is claimed only if the pre-registered pilot and fresh training gate pass.
