# ToolSandbox V8 code review

Status: reviewed after implementation; blocking issue resolved.

The independent same-family review found one blocking reproducibility issue:
V8 did not initially bind the V4.4 worker provider, model, base URL, thinking
mode, and horizon. The implementation now copies those fields from the frozen
V4.4 lock, checks the live environment before collection, verifies the CLI
model, and records the actual worker identity in the collection summary.

The review found no silent old-label/new-action mismatch and no official-score
or hidden-context leakage. A, B, and treatment-visible history must exactly
match the frozen V4.4 events before an old Rescue/Reverse label is attached.

Local assurance after the fix:

- V8 and V7 regression tests: 15 passed.
- Ruff: passed.
- Python compilation and `git diff --check`: passed.

The review is provisional because it used a same-family reviewer. The remote
run remains a feasibility experiment, not confirmatory evidence.
