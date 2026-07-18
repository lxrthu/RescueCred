# ToolSandbox signal audit code review

Date: 2026-07-18
Verdict: **DEPLOY YES**

The fresh review compared the implementation with pinned Apple ToolSandbox commit `165848b9a78cead7ca7fe7c89c688b58e6501219`.

## Resolved blockers

1. The original all-state-dependency selector could not reach 30 scenarios. The final selector deterministically prioritizes eligible state-dependency scenarios, then fills from the same single-user/multiple-tool/no-distraction contract. Static review found exactly 40 eligible non-RapidAPI scenarios.
2. Raw context tools exposed USER-only `end_conversation`. The adapter now applies the official `visible_to` AGENT filter for selection, schemas, and execution.
3. Natural coverage now counts replay-valid pairs only; continuation failures enter worker reliability accounting; official evaluator use is derived from branch provenance.
4. Snapshot integrity hashes databases/control fields plus a deterministic executable namespace fingerprint. Raw dill bytes are intentionally excluded because dill serialization is reversible but not canonical across equivalent console copies.
5. Worker reads have a configurable 180-second watchdog and terminate a stalled subprocess.

## Verified boundaries

- system bootstrap follows official `Scenario.play` semantics;
- tool calls go through official `ExecutionEnvironment.respond` and retain tool traces;
- branch score is official `EvaluationResult.similarity`;
- proposal/repair/continuation never receive milestones, minefields, references, evaluator values, or hidden databases;
- Python 3.9 grammar is valid.

Local main-agent validation: focused tests passed; full repository tests passed; Ruff and shell syntax checks passed.
