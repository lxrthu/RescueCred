# Claims from AppWorld Harness Results

- **verdict**: no
- **confidence**: high
- **review independence**: same-family
- **acceptance status**: provisional
- **integrity status**: unavailable; the remote JSON was user-pasted and locally transcribed

## Unsupported claim

The current evidence does not support the claim that the reference-free deployable Harness is reliable enough to authorize RescueCredit-v2 training on AppWorld. The final untouched 27-task holdout produced 23 correct repairs out of 34 changes (67.6% precision), below the frozen 90% floor.

## Supported limited findings

- AppWorld rollback and execution infrastructure works.
- The provenance-aware Harness can produce some correct repairs: 23/117 corrupt cases were rescued.
- No harm was observed among 75 clean cases in this holdout, but this is not a general safety guarantee.
- Adding visible candidate provenance improved development-set behavior, but no matched holdout ablation establishes a causal provenance benefit.

## Revised research claim

Automatic reference-free correction generation is separated from the core credit-assignment contribution. Future RescueCredit experiments may claim only conditional performance given a frozen externally supplied or independently verified correction source shared identically by all methods.

## Route

Stop tuning the current automatic Harness against observed development/holdout data. Freeze correction traces, then compare Mask+Correction and RescueCredit-v2 with matched models, budgets, seeds, and evaluation. Keep `test_normal` and `test_challenge` sealed until the method is frozen.

---

# Route-A Immediate-Effect Claim Verdict

- **verdict**: no
- **confidence**: high
- **review independence**: same-family
- **acceptance status**: provisional
- **integrity status**: unavailable; remote output was user-pasted and locally transcribed

The seed-42 deterministic AppWorld diagnostic does not support a causal-selection advantage for RescueCredit-v2 over Mask. Across 55 valid pairs, only one A/B pair changed the immediate official score; Mask and V2 had identical mean score, identical zero causal accuracy, and no wins or losses. Do not expand seeds under the same protocol.

The defensible claim is limited to: the infrastructure can replay paired branches without reference leakage, while comparative causal effectiveness remains unestablished. The next experiment must first create or identify sufficiently many informative disagreement events or use a leakage-safe bounded-horizon outcome.
