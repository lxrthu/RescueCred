# ToolSandbox V4 Independent Code Review

Date: 2026-07-18  
Final verdict: **DEPLOY YES**

## First review

The independent reviewer returned `DEPLOY NO` despite passing directed tests.
Blocking findings were:

1. snapshot exactness was a one-time success flag rather than an all-run audit;
2. natural Harness coverage alone could authorize zero-delta or harmful repairs;
3. the protocol accepted an incomplete source inventory and did not bind the
   Stage-0 gate or editable ToolSandbox runtime;
4. V4 could pass without a validated protocol lock;
5. intermediate official trace provenance, padding, and AUC were not independently
   recomputed;
6. the server preflight omitted the V4 protocol test.

Non-blocking findings requested structured visible exceptions, a non-ambiguous
`mean_terminal_delta`, delayed output-root creation, stronger sanity checks, and
an explicit development/fresh hash intersection check.

## Applied fixes

- Count every proposal snapshot and every A/B prefix restoration; require nonzero
  checks and zero mismatches.
- Split controlled mechanism and deployable Harness gates. Natural deployment now
  requires coverage, nonzero credit, rescues, an official outcome/progress rescue,
  wins over losses, and bounded harm.
- Require an exact frozen source inventory, plan hash, thresholds, model/provider/
  base URL/thinking mode, Stage-0 gate, disjoint scenario hashes, Python executable,
  complete ToolSandbox Python source-tree hash, vendor Git HEAD, and clean tracked
  vendor worktree.
- Make protocol validation and fresh scenario identity mandatory for every V4 gate.
- Validate every official score-trace point, reconstruct early-stop padding, and
  independently recompute terminal score and bounded progress AUC.
- Add the V4 protocol test to server preflight, delay output-root creation until
  preflight succeeds, and require a valid/provenance-safe sanity event.
- Pass structured visible error content plus `tool_call_exception` to natural repair.
- Separate terminal, outcome/progress, and efficiency-only result counts.

## Final review

The reviewer confirmed that the full ToolSandbox `*.py` tree hash, vendor Git
root/HEAD/clean-worktree checks, freeze/validate runtime identity, and non-entry
runtime tamper test are effective. Directed tests passed `28/28`; no blocking issue
remained.

Final verdict: **DEPLOY YES**.
