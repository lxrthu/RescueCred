# ToolSandbox V7 ActiveShadow feasibility plan

V6 found weak static signal (`ROC-AUC=0.621`) and zero Reverse recall under a
two-point Rescue budget. V7 tests whether one isolated A/B tool execution adds
the missing information. It does not retrain the policy and does not expose
reference actions, official score traces, full-trajectory outcomes, or hidden
state digests to the router.

The existing V4.4 raw audit already stores the first visible receipt from each
A/B branch. V7 hashes the two actions and first receipts into a bounded feature
vector. A task-grouped five-fold cross-fit trains (1) a static acquisition head
that chooses at most 30% of events to probe and (2) an ActiveShadow head that
routes probed events. Unprobed and low-confidence events abstain to correction B.

Frozen feasibility gates:

- ActiveShadow cross-task ROC-AUC >= 0.75;
- empirical Rescue drop <= 0.02;
- overall Reverse recall >= 0.20;
- probe rate <= 0.30;
- ActiveShadow AUC exceeds the static acquisition control.

Risk certification is reported separately. A one-sided exact binomial upper
bound at alpha=0.05 must be <=0.02 before any formal guarantee is claimed.
With zero errors this requires at least 149 independent Rescue calibration
events; the current 41-event Rescue subset can establish feasibility but cannot
certify the two-point bound.
