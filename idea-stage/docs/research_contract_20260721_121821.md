# Research Contract: RAPG / Runtime Action Substitution

## Core claims

1. Crediting executed B return to unexecuted proposal A creates replacement bias; Mask deletes intervention-conditioned proposal-gradient mass.
2. Ordinary repaired logs do not identify the missing A-return gradient.
3. Positive-propensity randomized Shadow plus RAPG identifies the local candidate-policy proposal gradient.
4. A positive efficiency claim requires at least 15% projected-gradient MSE reduction over uniform auditing at matched expected cost.
5. Autonomous policy improvement is not claimed unless a later fresh-task policy experiment passes.

## Pilot contract

- Ground truth: official paired ToolSandbox Full-Shadow returns.
- Split: deterministic task-group cross-fitting.
- Primary estimator: unclipped committed inverse propensity.
- Budget: fixed `c=1`, 20% expected audit rate, `p_min=0.05`.
- Stop before training on failed identity, bias, efficiency, or residual-only comparison gate.
