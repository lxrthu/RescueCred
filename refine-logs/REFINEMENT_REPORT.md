# RAPG Refinement Report

## Starting point

The initial AdaAudit idea used adaptive AIPW/HT sampling to lower counterfactual-return MSE. Review judged this direction faithful but too close to standard two-phase sampling to support an AAAI main algorithm claim.

## Key changes

- Recentered the method on the proposal/execution mismatch created by runtime action replacement.
- Defined a first-intervention local proposal-gradient estimand under fixed Harness occupancy.
- Added the replacement-bias, Mask-deletion, non-identification, and RAPG-identification theorem chain.
- Corrected conditional-unbiasedness and stochastic-Shadow variance statements.
- Restricted theory to a frozen behavior parameter and one local update.
- Separated the mathematical filtration from public learned-model features.
- Made finite-batch allocation implementable through a two-pass protocol.
- Added an exact behavior-policy identity gate; stale/off-policy banks cannot be relabeled on-policy post hoc.
- Restricted the main budget to expected cost with fixed `c=1`; no propensity clipping in primary results.
- Replaced a weak “CI contains zero” check with projected-gradient 5% bias equivalence.
- Made policy-level unassisted gain a mandatory empirical gate rather than a theorem overclaim.

## Score evolution

`7.23 -> 8.04 -> 8.54 -> 8.96 -> 9.10 (READY for Pilot 0)`

## Remaining evidence gap

No positive RAPG result exists yet. The next action is the 100--150-event identity-valid gradient-reconstruction Pilot 0. Only a passing result authorizes policy training or a positive paper claim.
