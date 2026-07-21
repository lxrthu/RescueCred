# RAPG: Randomized Proposal Credit behind Runtime Action Replacement

## Main contribution

When a Harness replaces proposal A with executed action B, the policy log-probability belongs to A while the observed return belongs to B. RAPG keeps safe B execution, randomly reveals a bounded subset of missing A returns through exact isolated Shadow, and uses committed propensities to recover the local proposal-policy gradient.

## Action-substitution theorem package

At frozen iteration `theta_k`, A is the full autoregressive tool-call sequence sampled from `pi_k`, `s_k=grad log pi_k(A|x)`, B is the Harness replacement, and `Y_A,Y_B` are bounded potential returns under the same fixed continuation. The target is the first-intervention, fixed-occupancy local gradient

```text
g_k = E_{x~d^{H,pi_k},A~pi_k}[s_k Y_A].
```

This is not claimed to equal the fully unassisted end-to-end gradient.

### 1. Replacement bias and Mask deletion

```text
g_Bcredit = g_k + E[s_k(Y_B-Y_A)]
Delta_mask = E[1{A replaced} s_k Y_A].
```

B-credit couples A's log-probability to the wrong potential return; Mask removes exactly the gradient mass associated with corrected proposals.

### 2. No-audit non-identification

If two latent environments induce the same repaired-log distribution `(x,A,B,Y_B,F)` but different `Y_A` and `g_k`, no observation-only estimator identifies `g_k` in both. Longer receipts or classifiers cannot recover a counterfactual erased by the observation interface.

### 3. RAPG identification

Let S be the mathematical pre-audit sigma-field and `F=phi(x,A,B,Y_B,public ledger metadata)` the only learned-model input. Snapshot/RNG contents are hash-bound and restored but never exposed as features. Commit `m(F)` and `p>0`, draw `Z|S~Bernoulli(p)`, and define

```text
Yhat_A = m(F) + Z/p (Y_A-m(F))
ghat = s_k * stopgrad(Yhat_A-b(x)).
```

For fixed potential outcomes, `E_Z[Yhat_A|S,Y_A]=Y_A` and `E[ghat]=g_k`. For stochastic Shadow, tower expectation identifies the conditional-mean target. The baseline is action-independent; B and Y_B enter only the augmentation model. The theorem covers one local on-policy update, not clipped/multi-epoch PPO.

## Budget mechanism

With `w=||s_k||`, `a=w^2 E[(Y_A-m(F))^2|S]`, and fixed pilot cost `c=1`, the allocation-dependent gradient second moment is `sum a/p`. Under a 20% expected audit budget the finite-batch interior solution is `p=lambda sqrt(a)`, clipped only by pre-registered `[p_min,1]`. A task-cross-fitted public-feature model predicts a. The primary estimator always uses the true committed `1/p`; clipped weights are a biased diagnostic only.

## Two-pass protocol

1. Freeze/hash policy, adapter, tokenizer, prompt, sampler, Harness, continuation, scorer, horizon, and `p_min`.
2. First pass: sample A from the exact policy, save pre-B snapshot/RNG commitment, execute safe B, seal Y_B, and compute public F, full-sequence LoRA score, and cross-fitted `m,ahat`.
3. Solve the batch dual lambda from sealed predictions only.
4. Bind all probabilities before opening any A outcome; independently draw audits.
5. Run isolated A Shadow only when selected; never feed it back to the live B trajectory.
6. Form RAPG, take one local update, and update residual models only for later batches.

## Pilot 0

Use 100--150 first-intervention Full-Shadow events. Existing data are valid only if A regenerates exactly under the bound checkpoint, adapter, tokenizer, prompt, sampler, schema, and seed; otherwise recollect a small clean bank. Post-hoc likelihood scoring is insufficient.

Across task-held-out folds and at least 1,000 fixed-propensity audit resamples, compare uniform, residual-only, score-only, RAPG `w*sigma`, and an unattainable oracle ceiling.

Pre-registered go gates:

- projected-gradient bias passes a 5% equivalence bound on fixed registered projections;
- held-out gradient MSE is at least 15% below uniform;
- RAPG beats residual-only;
- cosine similarity improves;
- gains are not task-family concentrated and pass ESS/max-weight checks;
- primary results use no propensity clipping.

Failure stops the method before policy training.

## Policy gate

Only after Pilot 0 passes, compare Mask + Correction, uniform RAPG, RAPG with gradient-aware allocation, and Full Shadow at matched cost. Primary outcome is fresh-task unassisted first-pass success; assisted safety is a non-inferiority constraint. Without gain over both Mask + Correction and uniform RAPG, no autonomous-improvement claim is made.

## Novelty boundary

Active importance sampling selects rewards from actions that were executed. RAPG handles a structurally different case: A was sampled but never executed, observed reward belongs to replacement B, and randomized Shadow creates the missing A-return observation channel. HT augmentation and square-root allocation are standard tools; the contribution is the action-substitution estimand, identification result, and policy consequence.
