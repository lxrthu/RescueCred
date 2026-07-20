# ToolSandbox V8 explicit visible-state probe

V7 showed that raw first-receipt hashing is insufficient. V8 replays the frozen
V4.4 treatment points for exactly one A/B step and records the state transition
visible to a deployed agent: appended visible history, tool exception/receipt,
and public tool-schema availability before and after execution. It never calls
the official evaluator and never exports hidden context/database state.

Candidate generation, scenario order, prefix advancement, A/B actions, and
event ids must reproduce the frozen V4.4 audit exactly. Any mismatch aborts the
run. V8 then reuses the V7 nested task cross-fit: disjoint model-training,
calibration, and untouched evaluation tasks in every outer fold.

Feasibility gates remain ROC-AUC >= 0.75, empirical Rescue drop <= 0.02,
Reverse recall >= 0.20, and probe rate <= 0.30. Formal two-point risk
certification remains disabled until a separate fixed-policy calibration set
contains enough Rescue events.
