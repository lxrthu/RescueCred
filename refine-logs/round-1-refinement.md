# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** A runtime Harness executes correction B instead of policy proposal A, so the assisted return identifies the executed system but not the credit attributable to the trainable proposal. We need a method that improves the unassisted policy under a bounded audit budget while deployment continues to default safely to B.
- **Must-solve bottleneck:** Rescue/Reverse direction is not reliably identifiable from deployment-visible static context, one-step receipts, two-step receipts, fixed state observers, or simple goal contracts. A method cannot stably learn an event-level label from an observation interface that erases the relevant counterfactual information.
- **Non-goals:** Do not tune another preference/RL loss on the same labels; do not require a perfect per-event A/B router; do not expose reference actions, official evaluators, or hidden database state to the deployed policy; do not claim a universally reliable automatic Harness.
- **Constraints:** Preserve safe default execution B; use exact isolated Shadow only on a budgeted subset; audit probabilities and predictors must be fixed before each draw; current banks are development-only and fresh task-disjoint confirmation is required; compute should fit the existing ToolSandbox/API-Bank pipeline and at most a small model pilot.
- **Success condition:** At the same Shadow interaction cost, the method has lower counterfactual-credit MSE than uniform auditing, remains unbiased under adaptive predictable sampling, and produces a measurable improvement in unassisted/first-pass policy behavior over Mask + Correction without reducing assisted safety.

## Anchor Check

- The revision still targets missing proposal credit caused by action replacement; it does not return to Rescue/Reverse routing.
- Exact Shadow creates the missing counterfactual only for randomly audited events. No hidden evaluator or reference action is exposed to the deployed policy.
- The theorem is deliberately scoped to the proposal objective under Harness-induced occupancy. Improvement after completely removing the Harness is an empirical outcome, not a theorem consequence.

## Simplicity Check

- **One dominant contribution:** replacement-aware policy-gradient estimation under runtime action substitution.
- **One supporting mechanism:** gradient-variance-optimal randomized Shadow auditing.
- **Removed from the core:** conformal routing, goal contracts, receipt classifiers, correction imitation, and multi-step probing.
- **New learned state:** one shared residual model with a mean head and a gradient-weighted second-moment head.

## Revised Proposal

# AdaAudit: Replacement-Aware Policy Gradients under Runtime Action Substitution

## 1. Technical gap

At the first path-changing intervention, a behavior policy samples proposal `A ~ pi_old(.|x)`, but the Harness executes `B = H(x,A)` and the environment reveals the assisted continuation. Assigning B's return to `log pi(A|x)` is generally biased: the log-probability belongs to A, while the observed outcome belongs to B. Masking the event discards precisely the errors the policy must learn from. Treating B as a preference label recreates the observed Rescue--Reverse tradeoff.

AdaAudit treats the unexecuted proposal return as a missing outcome and estimates the policy gradient directly, rather than first requiring a reliable event-level Rescue/Reverse classifier.

## 2. Estimand and scope

We define the **Harness-occupancy proposal objective**

```text
J_prop^H(theta) = E_{x ~ d^{H,pi_old}, A ~ pi_theta(.|x)}[Q_A^H(x,A)].
```

`d^{H,pi_old}` is the state distribution induced by the frozen behavior policy and fixed Harness before the first path-changing intervention. `Q_A^H(x,A)` is the bounded return obtained by restoring the exact pre-intervention snapshot, executing A once, then following a frozen continuation policy and fixed horizon. The primary theory and pilot use only the first path-changing intervention per trajectory, so no unsupported multi-intervention occupancy correction is hidden in the objective.

In deterministic replay environments, the snapshot and committed RNG state determine `G_A`. In stochastic environments, `G_A` is one unbiased draw from `Q_A^H`; repeated Shadow seeds are an optional variance study.

The unassisted end-to-end success of the updated policy is a primary empirical metric, but is not equated with `J_prop^H` in the theorem.

## 3. Commit-before-draw collection

For each eligible event, the system:

1. samples and hash-binds A from frozen `pi_old`;
2. obtains and hash-binds B from the fixed Harness;
3. safely executes B and seals its public trajectory and return `G_B`;
4. computes a stop-gradient outcome prediction `m_t = m_psi(x,A,B,G_B)` and audit probability `p_t` using only sealed pre-audit information;
5. commits feature, model-version, probability, and RNG hashes;
6. draws `Z_t ~ Bernoulli(p_t)`;
7. only if `Z_t=1`, restores the snapshot and runs exact isolated Shadow for A to observe `G_A`.

The safe B outcome is therefore a free control variate, but the audit decision cannot depend on `G_A`. Predictor and allocator parameters may update only after the current ledger entry is sealed.

## 4. Replacement-Aware Policy Gradient (RAPG)

Let

```text
s_t(theta) = grad_theta log pi_theta(A_t | x_t)
rho_t(theta) = pi_theta(A_t|x_t) / pi_old(A_t|x_t)
```

and let `b_t` be a pre-audit, action-independent policy baseline. Define the augmented proposal-return estimator

```text
Qhat_A = m_t + Z_t/p_t * (G_A - m_t),
```

and the replacement-aware gradient sample

```text
ghat_t(theta) = stopgrad(rho_t) * s_t(theta)
                * stopgrad(Qhat_A - b_t).
```

For a REINFORCE update, `rho=1` on-policy. For a one-epoch PPO-style update, `rho` is the usual behavior/current likelihood ratio; clipping is reported only as a biased optimization variant, not part of the unbiasedness theorem. `m`, `p`, the audit draw, return, and allocator are all stop-gradient.

**Theorem target 1 -- replacement-aware unbiasedness.** Conditional on the sealed event history, if `p_t>0` is predictable and Shadow produces an unbiased draw of `Q_A^H`, then `E[Qhat_A | x,A,B,G_B] = Q_A^H(x,A)`. Consequently, the on-policy RAPG sample is unbiased for the score-function gradient of `J_prop^H`; the importance-weighted form is unbiased for the corresponding fixed-occupancy off-policy objective when its support condition holds.

This is the paper's central estimator result. It formalizes the proposal/execution mismatch created specifically by runtime action replacement.

## 5. Gradient-variance-optimal auditing

Outcome-MSE-optimal sampling is not enough. The relevant error is the second moment of the stochastic policy-gradient residual. Define

```text
w_t = ||rho_t s_t||_2
sigma_t^2 = E[(G_A - m_t)^2 | sealed pre-audit information]
c_t = expected Shadow cost
a_t = w_t^2 sigma_t^2.
```

Under the expected cost budget `sum_t p_t c_t <= C`, the audit-dependent trace of gradient covariance is

```text
sum_t (1/p_t - 1) a_t.
```

The relaxed oracle allocation is

```text
p_t* = clip(lambda * w_t sigma_t / sqrt(c_t), p_min, 1),
```

with `lambda` chosen to meet the expected budget. Thus an event is audited because its missing outcome can materially change the policy gradient, not merely because its return is uncertain.

For LoRA/adapters, `w_t` is approximated by a per-example adapter-score or diagonal-Fisher sketch; the full foundation-model gradient is never stored. A shared lightweight residual model predicts `m_t` and the second moment `a_t` from task-disjoint training audits.

**Theorem target 2 -- oracle allocation.** Among predictable Bernoulli allocations satisfying the same expected cost budget and probability floor, the expression above is minimized by the clipped square-root allocation.

**Theorem target 3 -- plug-in robustness.** Let `ahat_t` satisfy `a_t/r <= ahat_t <= r a_t` for all unclipped events and let the plug-in and oracle allocations use the same expected budget. The plug-in allocation's audit-dependent gradient second moment is at most an `r^2` multiplicative factor of the oracle objective, up to explicit floor/clipping boundary terms. The paper will state the exact boundary term rather than call a learned allocator globally budget-optimal.

The theory uses an expected Shadow-cost budget. Realized cost and tail probability are reported. A hard-cap allocator is an engineering ablation and carries no unbiasedness claim after the cap becomes history-dependent unless its conditional propensities remain recorded and positive.

## 6. Supporting identifiability statement

If two latent environments induce the same distribution over the repaired log `(x,A,B,G_B)` but different `Q_A^H`, no observation-only estimator can identify proposal credit in both. Randomized Shadow changes the observation process by revealing `G_A` with known positive propensity. This proposition motivates intervention but is supporting, not a parallel headline contribution.

## 7. Algorithm block

```text
Input: frozen pi_old, Harness H, residual model m/a, expected budget C
for each trajectory:
    run until first path-changing Harness intervention
    seal x, A~pi_old, B=H(x,A), execute safe B, seal G_B
    compute score sketch w and predicted ahat; choose committed p
    draw Z~Bernoulli(p)
    if Z=1: exact Shadow A from snapshot -> G_A
    Qhat <- m + Z/p * (G_A-m)       # Z=0 term is exactly zero
    store (x,A,pi_old(A|x),Qhat,b,p,c)
update policy with RAPG/PPO surrogate; never backprop through m, p, or Qhat
update residual model only from previously sealed audited records
```

## 8. Novelty boundary

The adaptive allocation alone is not claimed as novel: gradient-norm importance sampling and active importance sampling for policy gradients already exist. The new object is a policy-gradient estimator for a runtime **proposal/execution split**, where the behavior policy sampled A, a Harness replaced it with B, B's return is observed freely, and A's return is available only through randomized exact intervention. The allocator is derived for the variance of this replacement-aware gradient and uses B as a control variate.

## 9. Claim-driven validation

### Gate A: estimator and allocation simulation

Use existing exact Full-Shadow records only as a hidden outcome bank. Stream them task-disjointly and reveal `G_A` according to committed probabilities.

- Baselines: uniform Bernoulli audit, return-residual allocation without score norm, uncertainty sampling without propensity correction, full Shadow.
- Budgets: 5%, 10%, 20%, 30% matched Shadow cost.
- Primary: gradient-vector MSE or projected-gradient MSE versus Full Shadow.
- Integrity: bias/coverage, propensity support, effective sample size, realized cost.
- Continue only if RAPG is statistically indistinguishable from zero bias and reduces gradient MSE versus uniform at matched cost on held-out tasks.

### Gate B: small policy pilot

Train one small policy/adapter using Mask + Correction, uniform RAPG, AdaAudit RAPG, and Full Shadow. Freeze all choices before evaluation.

- Primary: unassisted first-pass task success on fresh tasks.
- Secondary: assisted success non-inferiority, proposal validity, audit cost, gradient variance.
- Required positive result: AdaAudit beats uniform audit at matched Shadow cost and Mask + Correction, approaches Full Shadow, and does not reduce assisted safety.

### Gate C: fresh confirmation

Use a task-disjoint bank never used for V6--V9, DeltaGuard, Goal Contract, residual training, allocation tuning, or policy selection. Report task-cluster bootstrap intervals and all failures.

## 10. Falsification and stopping rules

- If gradient-weighted allocation does not beat uniform in Gate A, stop; the main efficiency claim fails.
- If unbiased credit lowers gradient MSE but does not improve unassisted behavior in Gate B, retain an estimator result only and do not claim agent improvement.
- If gains disappear on fresh tasks, present the existing Compensation Trap and non-identifiability findings as the empirical contribution, not AdaAudit as a solved method.

## 11. Paper contribution statement

1. We formalize runtime action replacement as a missing-proposal-return policy-gradient problem.
2. We derive RAPG, an unbiased estimator using committed randomized Shadow audits and the executed B trajectory as a control variate.
3. We derive gradient-variance-optimal audit allocation and a plug-in robustness bound under a fixed expected cost budget.
4. We test whether the resulting credit estimator improves unassisted agent behavior at matched intervention cost; the prior routing failures serve as evidence that observational classification is insufficient, not as another algorithm branch.
