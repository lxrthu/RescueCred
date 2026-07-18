# ToolSandbox V4.1 Tool-ID Harness Plan

## Motivation

The frozen V4 fresh audit validated scenario identity and snapshots but produced
39 invalid proposals in 40 distraction-tool scenarios. This is a deployable
Harness coverage failure, not evidence about the lexicographic credit rule.

## Frozen method change

V4.1 changes only the public Harness interface:

1. Sort deployment-visible public tool schemas by exact public function name.
2. Assign deterministic opaque identifiers `T0000`, `T0001`, ... .
3. Show the model the identifier, public name, description, and parameters.
4. Require `{"tool_id":"T0000","arguments":{...}}`.
5. Map the identifier back to the exact public name in trusted code before
   execution.

No milestone, minefield, reference action, evaluator score, hidden state, or
protected value enters the mapping or prompt. V4 credit, thresholds, official
scoring, and snapshot logic are unchanged.

## Frozen partitions

- Previously observed and excluded: old V4 development hashes plus offset
  40–79 fresh hashes, read from the old frozen protocol lock.
- Diagnostic: offset 80, limit 5, seed 42, H4, search depth 4.
- Confirmatory fresh audit: offset 85, limit 40, seed 42, H8, search depth 8.
- Both diagnostic and confirmatory protocol locks are frozen before diagnostic
  outcomes. The confirmatory lock excludes both old V4 and diagnostic hashes.

If fewer than 5 diagnostic or 30 confirmatory scenarios exist, stop before any
outcome. Do not reuse observed scenarios.

## Diagnostic hard gate

All must pass:

- exactly 5 selected scenarios;
- at least 4 scenarios produce a public-schema-complete proposal;
- schema-valid proposal coverage at least 0.80;
- at least 3 valid controlled paired events;
- worker request failures at most 10%;
- exact snapshot/prefix restoration;
- official evaluator and protocol validation pass.

The confirmatory audit cannot start when this gate fails.

## Confirmatory gate

Use the unchanged V4 mechanism and deployable-Harness thresholds. Report
terminal, progress, and efficiency-only credit separately. Do not train unless
the frozen confirmatory gate passes.
