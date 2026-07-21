# RAPG Review Summary

Five strict review rounds moved the proposal from adaptive outcome sampling (7.23/10) to an action-substitution policy-gradient method ready for a falsification pilot (9.10/10).

## Final reviewer judgment

RAPG is a credible AAAI main-contribution candidate only as a complete action-substitution package:

1. runtime replacement creates explicit B-credit bias;
2. Mask deletes intervention-conditioned proposal-gradient mass;
3. ordinary repaired logs do not identify the missing proposal gradient;
4. randomized exact Shadow creates the missing A-return channel;
5. propensity-corrected RAPG identifies the local proposal gradient;
6. gradient-aware allocation is a budget mechanism, not a separate novelty claim.

READY means ready to implement Pilot 0. No positive estimator-efficiency or policy-improvement result has yet been established.

## Frozen evidence gate

Proceed to policy training only if the identity-valid Pilot 0 passes projected-gradient bias equivalence, reduces held-out gradient MSE by at least 15% versus uniform auditing, beats residual-only allocation, and passes ESS/task-concentration checks without propensity clipping.
