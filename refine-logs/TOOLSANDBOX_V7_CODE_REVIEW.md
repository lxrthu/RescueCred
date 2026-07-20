# ToolSandbox V7 ActiveShadow code review

Status: deployment-ready for exploratory feasibility; not ready for a formal
risk guarantee or a single-policy confirmation claim.

The first independent review found two blockers: OOF labels were reused for
calibration/threshold selection and final pipeline metrics, and a pointwise
Clopper-Pearson bound was treated as valid after scanning thresholds on the
same sample. The implementation now uses nested task cross-fitting. Every outer
fold has disjoint model-training, calibration, and untouched evaluation tasks;
only outer evaluation rows contribute to the feasibility gate. Threshold-scan
risk certification is disabled, the saved checkpoint is marked
`deployment_ready=false`, and the aggregate binomial upper bound is explicitly
descriptive and uncertified.

The second review confirmed the nested split and risk-claim fixes, then found
that protected-key rejection covered receipt content but not exceptions. Both
content and exception payloads are now recursively scanned case-insensitively
for label, reward, reference, official-score, score-trace, progress, and branch
fields before feature construction. Tests cover protected fields in both
locations.

Local verification: V7/V6/V5 tests pass except Torch-dependent tests skipped in
the local Windows environment; Python compilation, Ruff, and `git diff --check`
pass. Full Torch execution remains a server-side sanity step.
