# ToolSandbox V9 offline two-step ActiveShadow

V9 tests the cheapest no-recollection extension after V7's one-step receipt
signal failed. It reads only the first two executed actions and visible receipts
from each frozen V4.4 A/B branch. Third and later receipts, official scores,
score traces, returns, decisions, and ending context digests are prohibited as
features.

The V7 nested task cross-fit, calibration, abstention rule, Rescue budget, and
probe budget remain unchanged. The frozen feasibility gate requires ROC-AUC at
least 0.75, empirical Rescue drop at most 0.02, Reverse recall at least 0.20,
probe rate at most 0.30, improvement over the static acquisition representation,
and improvement over V7's one-step ActiveShadow AUC.

This is an offline feasibility test. A pass supports implementing a live
two-step probe; it is not itself deployment or formal risk-certification
evidence.

Deployment validity requires invoking the same frozen continuation policy once
inside each isolated A/B branch after receipt 1. Thus every probed event costs
two attempted continuation-policy calls and at most four tool executions total;
if a branch stops after receipt 1, its second execution is absent. The 30% probe-
rate budget is an event budget, not a compute-cost guarantee; attempted and
maximum rates are reported.
