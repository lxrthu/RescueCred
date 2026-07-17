# Initial Experiment Results

## M0 deterministic sanity — PASSED

- 23 unit tests passed after the integrated trajectory/shadow rewrite.
- Snapshot replay determinism, audit commit ordering, provenance masking, estimator convergence, verifier scope and budget identity are covered.
- Rescue-MDP produced 45 exact `(state, action, condition)` Q records.

## M1 API-Bank controlled infrastructure — PASSED

- Official raw dialogues found: 264.
- Executable after deterministic filter: 254.
- Frozen train/dev/test_id/test_tool_ood: 138/19/23/29; 45 mixed-tool conflicts were excluded.
- Normalized-goal, exact-action, composite-family and atomic-tool overlap counters are all zero.
- Injected infrastructure smoke: 114 episodes; replay failure rate 0.
- Smoke results are explicitly `not_research_evidence=true`.

## M2-M4 — NOT RUN LOCALLY

No H200 training was performed in this local Windows session. Therefore there is no claim that RescueCredit improves policy ability. Run the cloud pilot and obey `outputs/pilot/gate.json` before any confirmatory claim.

## Readiness

- Engineering infrastructure: ready for cloud pilot.
- Research evidence: insufficient until real model pilot and confirmatory runs finish.
- Ready for result-to-claim: NO.
