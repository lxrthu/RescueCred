# Research Proposal: AdaAudit — Budget-Optimal Interventional Credit for Harnessed Agents

## Problem Anchor

- **Bottom-line problem:** A runtime Harness executes correction B instead of policy proposal A, so the assisted return identifies the executed system but not the credit attributable to the trainable proposal. We need a method that improves the unassisted policy under a bounded audit budget while deployment continues to default safely to B.
- **Must-solve bottleneck:** Rescue/Reverse direction is not reliably identifiable from deployment-visible static context, one-step receipts, two-step receipts, fixed state observers, or simple goal contracts. A method cannot stably learn an event-level label from an observation interface that erases the relevant counterfactual information.
- **Non-goals:** Do not tune another preference/RL loss on the same labels; do not require a perfect per-event A/B router; do not expose reference actions, official evaluators, or hidden database state to the deployed policy; do not claim a universally reliable automatic Harness.
- **Constraints:** Preserve safe default execution B; use exact isolated Shadow only on a budgeted subset; audit probabilities and predictors must be fixed before each draw; current banks are development-only and fresh task-disjoint confirmation is required; compute should fit the existing ToolSandbox/API-Bank pipeline and at most a small model pilot.
- **Success condition:** At the same Shadow interaction cost, the method has lower counterfactual-credit MSE than uniform auditing, remains unbiased under adaptive predictable sampling, and produces a measurable improvement in unassisted/first-pass policy behavior over Mask + Correction without reducing assisted safety.

## Technical Gap

Runtime repair creates a missing-counterfactual problem. The log contains the return of B, while policy learning needs the return that would have followed A. Existing RescueCredit experiments establish that exact paired Shadow can recover this label, but static, one-step, two-step, and hand-compiled public evidence do not predict it robustly across tasks. Therefore the missing quantity is not primarily a representation-learning target; it is an identification target.

The smallest adequate intervention is not a larger classifier. It is randomized counterfactual auditing. A uniform audit is unbiased but wastes budget on predictable or low-variance events. The missing algorithmic piece is to allocate audit probability adaptively using only pre-outcome information, while retaining an unbiased credit estimator despite this nonuniform allocation.

Two routes were considered:

- **Route A — adaptive randomized credit auditing:** choose a predictable per-event audit probability and use an augmented inverse-propensity estimator. This directly creates the missing information and admits an unbiasedness/variance theorem.
- **Route B — LLM-generated discriminating probes:** synthesize a public query intended to separate A/B. This remains dependent on whether the deployed tool interface contains a separating query, and current DeltaGuard/Goal Contract results show no reliable witness. It is retained only as future work.

Route A is chosen because it solves the anchored identification problem rather than continuing to guess from an insufficient observation interface.

## Method Thesis

- **One-sentence thesis:** AdaAudit restores proposal credit behind runtime correction by predictably allocating a limited number of isolated counterfactual executions and applying an adaptive doubly robust estimator whose audit probabilities are optimized for variance per unit cost.
- **Why this is the smallest adequate intervention:** It adds one predictable audit allocator and one scalar outcome predictor; policy, Harness, environment, and correction generator remain frozen during credit collection.
- **Why timely:** Recent verifier-based and intervention-based agent training exploits executable feedback, but does not address the proposal/execution mismatch created when a runtime system replaces the action before reward is observed.

## Contribution Focus

- **Dominant contribution:** Budget-optimal, sequentially unbiased counterfactual credit estimation for policies operating behind runtime action replacement.
- **Supporting contribution:** An observational non-identifiability result showing why post-repair logs and increasingly long unstructured receipts cannot generally identify proposal credit without an intervention or a separating observation.
- **Explicit non-contributions:** A new foundation model, a new Harness generator, a universal deployment router, or a new benchmark.

## Proposed Method

### Complexity Budget

- **Frozen/reused:** policy architecture, Harness, continuation policy, exact snapshot/restore, existing Shadow scorer, and standard policy optimizer.
- **New trainable components:** (1) outcome mean predictor `mu(x)`; (2) nonnegative residual-scale predictor `sigma(x)` used only for audit allocation. They may share one lightweight encoder/head.
- **Excluded:** learned Rescue/Reverse router, longer receipt horizon, multi-agent verifier stack, and online modification of B.

### System Overview

```text
state/history x -> policy proposes A -> Harness proposes B -> execute B safely
        |                 |
        | pre-outcome     +-> predictable allocator p_t(x,A,B,cost)
        |                            |
        +----------------------------+-- Bernoulli audit Z_t
                                      |
                       if Z_t=1: isolated Shadow executes A -> G_A
                       if Z_t=0: no A outcome is consumed

mu_t(x,A) + Z_t/p_t * (G_A-mu_t) -> unbiased proposal credit G_A_hat
G_A_hat -> proposal-token advantage; G_B -> executed-system logging only
audited residual -> update mu/sigma after current event
```

### Core Mechanism

For intervention event `t`, let `x_t` contain only information available before the audit draw, `G^A_t` be the isolated return of proposal A, `mu_t(x_t)` a predictor trained only on earlier audited events, and `p_t` the committed audit probability. Draw `Z_t ~ Bernoulli(p_t)` and define

```text
Ghat^A_t = mu_t(x_t) + Z_t / p_t * (G^A_t - mu_t(x_t)).
```

Because `mu_t` and `p_t` are predictable, `E[Ghat^A_t | history, x_t] = G^A_t`. The estimator remains unbiased even though difficult events receive larger audit probabilities.

Under expected audit-cost budget `sum p_t c_t <= C`, conditional variance is proportional to `(1/p_t - 1) E[(G^A_t-mu_t)^2 | x_t]`. The relaxed optimum is

```text
p*_t = clip(lambda * sigma_t(x_t) / sqrt(c_t), p_min, 1),
```

where `lambda` is chosen to satisfy the budget. This is Neyman-style allocation over intervention events. A budget accountant commits `p_t`, predictor version, feature hash, and RNG commitment before the draw. Predictors update only after the current audit record is sealed.

Proposal tokens receive an advantage constructed from `Ghat^A_t`; Harness/tool/environment tokens receive zero policy gradient. Correction imitation may remain a separately weighted auxiliary objective, but it is not the main contribution and is deleted if it recreates Rescue/Reverse interference.

### Supporting Identification Result

Let `O` be the deployed post-repair log. If two latent worlds agree on `P(O | x,A,B)` but disagree on `G^A`, then no estimator measurable with respect to `O` can identify proposal credit in both worlds. Static routers, receipt encoders, and longer receipt horizons remain inside this limitation unless they add a separating observation. Randomized isolated execution changes the observed sigma-algebra and identifies `G^A` on audited events; inverse-propensity correction transfers this information to the stream-level credit estimate.

### Modern Primitive Usage

- A frozen LLM encoder may supply features for `mu/sigma`, but it is an outcome model, not a judge or ground-truth source.
- Executable Shadow is the verifier and source of actual counterfactual outcomes.
- Online conformal/e-process risk control is optional only for reporting or stopping; it is not required for the core unbiasedness claim.

### Training Plan

1. Warm-start `mu/sigma` from training-only exact Shadow records using task-disjoint folds.
2. Freeze initial predictor and allocator configuration.
3. During collection, commit `p_t` before `Z_t`; update predictors only after sealing audited outcome.
4. Train policy with proposal-token advantage from `Ghat^A`; compare with Mask, uniform randomized audit, and Full Shadow.
5. Freeze method before fresh-task confirmation. Development banks already inspected are excluded from confirmation claims.

### Failure Modes and Diagnostics

- **Poor `mu`:** unbiasedness remains, but variance rises. Detect via audited residual calibration and compare to uniform allocation.
- **Tiny probabilities:** control with frozen `p_min`, weight clipping only as a separately reported biased ablation, and effective sample size.
- **Replay invalidity:** exclude before outcome use and report cost; no imputation.
- **Harness unreliable:** state claims conditional on a fixed externally supplied/verified correction source shared by all methods.
- **No autonomous improvement:** the estimator contribution may still hold, but the paper cannot claim policy improvement; stop before confirmation-scale training.

## Novelty and Elegance Argument

Counterfactual credit assignment, shielded RL, and interactive imitation each provide nearby ingredients, but they generally assume the learner observes its executed action or receives an expert intervention label. AdaAudit targets the proposal/execution split itself: reward comes from B while the trainable log-probability belongs to A. Its novelty is the predictable, budget-optimal audit allocation plus unbiased augmented estimator at this runtime-repair boundary. The negative routing results establish why this intervention is necessary rather than decorative.

## Claim-Driven Validation Sketch

### Claim 1: AdaAudit identifies proposal credit at lower cost

- **Minimal experiment:** replay the frozen exact ToolSandbox/API-Bank Shadow bank as a sequential stream with task-disjoint cross-fitting.
- **Baselines:** no audit predictor, uniform Bernoulli audit, uncertainty sampling without propensity correction, Full Shadow.
- **Metrics:** bias, MSE, confidence interval coverage, effective sample size, sign/rank accuracy, Shadow steps.
- **Decisive evidence:** near-zero bias and lower MSE than uniform auditing at 5%, 10%, 20%, and 30% matched cost.

### Claim 2: Better credit improves the unassisted policy

- **Minimal experiment:** one fixed verified-correction training stream, Mask + Correction versus uniform audit versus AdaAudit versus Full Shadow.
- **Metrics:** unassisted/first-pass task success primary; assisted success non-inferiority; audit cost.
- **Decisive evidence:** AdaAudit improves unassisted behavior over Mask and uniform audit, approaches Full Shadow, and retains assisted safety.

### Claim 3: Intervention is necessary

- **Minimal experiment:** summarize frozen V6/V7/V9/DeltaGuard/Goal Contract boundaries and verify the observational-equivalence construction in a controlled MDP.
- **Decisive evidence:** observation-only methods fail on paired indistinguishable worlds, while randomized audit remains unbiased.

## Experiment Handoff Inputs

- **Must prove:** predictable unbiasedness; variance-optimal allocation under budget; autonomous-policy gain at matched Shadow cost.
- **Must-run ablations:** uniform versus adaptive `p`; direct IPW versus augmented estimator; frozen versus online-updated predictor; no correction-preference auxiliary.
- **Critical datasets:** exact controlled MDP, development-only ToolSandbox paired bank for estimator simulation, fresh task-disjoint controlled agent stream for training confirmation.
- **Highest-risk assumption:** predictor uncertainty ranks residual magnitude well enough to improve variance; autonomous policy optimization can use noisy unbiased credit without instability.

## Compute & Timeline Estimate

- Estimator simulation and theorem checks: CPU, less than one day.
- Small policy pilot: one to two GPUs, roughly 6–12 GPU-hours.
- Fresh confirmation only after the one-seed gate passes.
