# ToolSandbox Signal Audit Plan (frozen before outcomes)

Date: 2026-07-18

## Research question

Does a stateful benchmark with official per-turn snapshots and milestone-DAG scoring expose denser causal credit for Harness interventions than the sparse AppWorld controlled-state diagnostic?

## Frozen sequence

1. Keep RescueCredit V3.1 unchanged.
2. Pin Apple ToolSandbox at commit `165848b9a78cead7ca7fe7c89c688b58e6501219`.
3. Verify exact snapshot restoration, public schemas, and official evaluator availability.
4. Run a 3-scenario integration smoke.
5. Run 40 single-user, multiple-tool, no-distraction scenarios at horizon 8, deterministically prioritizing every eligible state-dependency scenario.
6. Report natural visible-error repair separately from controlled missing-required-argument signal.
7. Promote ToolSandbox to a V3.1 method comparison only if the frozen signal gate passes.

## Primary signal gate

- at least 30 selected scenarios;
- at least 20 valid controlled paired events;
- controlled nonzero causal rate at least 20%;
- at least 3 natural visible-error repair events;
- worker failure rate at most 10%;
- exact snapshot restoration and official ToolSandbox evaluation.

The 20% target is a minimum density threshold, not a claimed performance gain. A passing audit does not establish autonomous task-success improvement.

## Reference boundary

Proposal, repair, and continuation workers receive only visible messages, public tool schemas, visible receipts, and proposal A. ToolSandbox milestones, minefields, reference traces, hidden databases, and evaluator outputs are excluded. Official `EvaluationResult.similarity` is read only after each branch terminates.
