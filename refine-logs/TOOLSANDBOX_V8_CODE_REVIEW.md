# ToolSandbox V8 code review

Status: reviewed after implementation; blocking issue resolved.

The independent same-family review initially found one blocking reproducibility
issue: the live worker identity was not fully bound. Those source fields were
bound, but the first remote run then exposed a more fundamental problem: LLM
candidate generation is not bitwise reproducible, so a newly generated B
drifted from the frozen V4.4 candidate.

The corrected collector makes no live worker requests. It binds the original
worker identity for provenance, loads the frozen A/B actions, deterministically
replays each prefix from frozen agent-visible history, and requires exact
history and public-schema agreement before probing A/B for one step.

The review found no silent old-label/new-action mismatch and no official-score
or hidden-context leakage. A, B, and treatment-visible history must exactly
match the frozen V4.4 events before an old Rescue/Reverse label is attached.

Local assurance after the fix:

- V8 and V7 regression tests: 17 passed after the deterministic replay fix.
- Ruff: passed.
- Python compilation and `git diff --check`: passed.

The review is provisional because it used a same-family reviewer. The remote
run remains a feasibility experiment, not confirmatory evidence.
