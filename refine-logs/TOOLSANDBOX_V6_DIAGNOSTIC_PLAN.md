# ToolSandbox V6 Reverse-only diagnostic

V5 proved that an unconstrained KEEP/FLIP router can improve Reverse while
damaging Rescue. V6 does not update the base policy or Mask adapter. It reuses
the frozen public-only V5 feature cache and asks a narrower question: can a
task-grouped diagnostic distinguish `reverse_preference` from
`rescue_preference` using only information available before branch outcomes?

Two probes are trained with identical task-grouped five-fold splits. The margin
probe is a confidence-only control. The semantic probe is a regularized MLP over
the frozen A/B representation difference and margin. Out-of-fold scores are
Platt calibrated. At deployment the router always chooses Harness correction B
unless calibrated Reverse probability exceeds a frozen threshold; all other
events abstain to B. Threshold selection maximizes Reverse recall subject to a
2-point Rescue non-inferiority budget.

Primary diagnostics are cross-task ROC-AUC, PR-AUC lift over prevalence, and
Reverse recall at the Rescue budget. Known development and V4.5 confirmation
sets are consistency/post-hoc checks only. A passing diagnostic requires
semantic ROC-AUC at least 0.70, PR-AUC lift at least 0.10, improvement over the
margin control, and at least 0.10 Reverse recall without exceeding the frozen
Rescue budget. A fresh scenario profile is required for a confirmatory claim.
