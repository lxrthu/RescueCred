# ToolSandbox receipt-horizon paired statistical audit

This final no-recollection analysis compares V7 one-step and V9 two-step OOF
predictions on exactly the same frozen events. It resamples whole task groups,
never individual events, preserving within-task dependence.

The protocol freezes 20,000 task bootstrap replicates and 20,000 paired
task-swap permutations at seed 42. Primary inference is the V9-minus-V7
cross-task ROC-AUC delta. Reverse recall, Rescue drop, and probe-rate deltas are
secondary operating-point descriptions.

This audit cannot reopen model selection or support a broader impossibility
claim. It only determines whether extending the tested receipt representation
from one to two steps improves, worsens, or does not significantly change the
paired frozen-task result.

Intervals and permutation tests are conditional on the frozen OOF predictions.
They preserve task clustering but do not include model-retraining uncertainty
or justify population-wide conclusions. Secondary operating-point p-values are
exploratory and receive no multiplicity correction.
