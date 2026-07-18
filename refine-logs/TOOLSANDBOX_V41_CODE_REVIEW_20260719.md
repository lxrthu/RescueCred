# ToolSandbox V4.1 Tool-ID Harness Code Review

**Review mode:** local-only (secondary-agent delegation unavailable under the
active execution policy)

## Verdict

DEPLOY YES for the five-scenario diagnostic only. The fresh 40-scenario audit
is automatically blocked unless the frozen diagnostic gate passes.

## Blocking checklist

- Reference boundary: PASS. IDs are constructed only from deployment-visible
  public function names, descriptions, and parameter schemas.
- Deterministic mapping: PASS. Public functions are sorted by exact name before
  assigning `T0000`, `T0001`, ...; trusted code maps IDs back to exact names.
- Schema-complete actions: PASS. Required fields must be present and unknown
  fields are rejected; recursive JSON type validity is not claimed.
- Error observability: PASS. Proposal and repair error types are counted without
  exporting prompts, secrets, protected labels, or hidden state.
- Previously observed scenario exclusion: PASS. Old development and offset
  40–79 hashes are excluded. Evaluation rechecks excluded protocol file hashes,
  embedded scenario hashes, and the fresh intersection.
- Pre-outcome freeze: PASS. Diagnostic and confirmatory locks are created before
  the first diagnostic model request.
- Sanity-first gate: PASS. The runner cannot reach offset 85 unless the five-row
  Tool-ID diagnostic meets coverage, paired-event, worker, snapshot, evaluator,
  and protocol thresholds.
- Outcome metric: PASS. The unchanged official ToolSandbox evaluator scores both
  branches; no model output is used as ground truth.

## Validation

- Directed V4/V4.1 ToolSandbox tests: 24 passed.
- Full repository suite: passed with 2 intentional skips.
- Ruff on every touched Python file: passed.
- Both cloud-runner Bash syntax and Git whitespace checks: passed.

## Remaining uncertainty

The local environment does not contain the pinned ToolSandbox runtime or the
configured remote LLM. Therefore deployment is authorized only for the frozen
five-scenario diagnostic; its hard gate decides whether the fresh audit runs.
