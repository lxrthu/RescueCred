# ToolSandbox V4.5 experiment code review

Date: 2026-07-20
Review mode: local-only (secondary-agent review unavailable under the active collaboration constraint)

## Scope

- Frozen offset-125 development and offset-165 confirmation candidate protocols
- Frozen matched Mask/V4.5 anchored-learner protocol
- Same-distribution candidate audit, data preparation, training, evaluation, and gates
- Server runner and regression tests

## Blocking checklist

- Method matches plan: PASS. Mask always uses B>A; V4.5 follows frozen bidirectional Shadow direction.
- Same data and budget: PASS. Both methods use the same deterministic 126-presentation sequence for three epochs.
- Evaluation against official outcomes: PASS. Candidate actions are fixed before official ToolSandbox branch outcomes are joined offline.
- Split leakage: PASS. Offset 85 trains, offset 125 develops, offset 165 confirms, and offset 205 is untouched.
- Confirmation protected by development gate: PASS.
- Private outcome fields absent from model prompts/training rows: PASS.
- Source, model, protocol, adapter, result, and event identities bound: PASS.
- Seeds and output paths explicit: PASS.
- Sanity/regression tests: PASS, full local suite 253 passed and 2 skipped.

## Fix made during review

The shared V4.3/V4.5 trainer initially emitted the V4.3 sampling identity for a
V4.5 protocol. This would have caused a frozen-config rejection before
training. Sampling identity is now selected from the protocol status and is
covered by the V4.5 tests.

## Remaining operational boundary

Local Windows lacks the remote ToolSandbox runtime artifacts, GPUs, and frozen
V4.4 outputs, so the full protocol-freeze and GPU sanity must run on the
server. The runner performs compilation, focused tests, LLM reachability, and
all frozen-artifact preflights before spending GPU time.
