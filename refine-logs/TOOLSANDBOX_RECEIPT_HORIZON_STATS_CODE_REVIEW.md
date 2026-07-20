# Receipt-horizon statistical audit code review

Status: provisional same-family review; blocking issues resolved.

The reviewer confirmed that whole-task bootstrap multiplicities, paired
task-level method swaps, nonlinear metric recomputation, permutation tails, and
Monte Carlo corrections are correct.

Two blockers were fixed before deployment. Gate integrity now requires the
complete expected nonempty key set, rather than allowing an empty dictionary to
pass. A positive routing result now requires statistical AUC improvement plus
all original absolute gates: AUC at least 0.75, Reverse recall at least 0.20,
Rescue drop at most 0.02, and probe rate at most 0.30.

The audit additionally verifies routed implies probed, requires 126 paired
events, binds the task set, removes the outcome-dependent precheck, and labels
secondary p-values exploratory without multiplicity correction. Inference is
explicitly conditional on frozen OOF predictions and excludes retraining and
population uncertainty.

The single re-review reported no remaining blocking issues.
