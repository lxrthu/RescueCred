# Initial Experiment Results

## M0 deterministic sanity — PASSED

- 26 unit tests passed after the integrated trajectory/shadow rewrite and post-review hardening.
- Snapshot replay determinism, audit commit ordering, provenance masking, estimator convergence, verifier scope and budget identity are covered.
- Rescue-MDP produced 45 exact `(state, action, condition)` Q records.

## M1 API-Bank controlled infrastructure — PASSED

- Official raw dialogues found: 264.
- Executable after deterministic filter: 254.
- Frozen train/dev/test_id/test_tool_ood: 138/19/23/29; 45 mixed-tool conflicts were excluded.
- Normalized-goal, exact-action, composite-family and atomic-tool overlap counters are all zero.
- Injected infrastructure smoke: 114 episodes; replay failure rate 0.
- Deterministic delayed-recovery support gate: 10 eligible tasks, 10 zero and 10 successful counterfactual returns, `Var(G0)=0.25`.
- Smoke results are explicitly `not_research_evidence=true`.

## M2-M4 — NOT RUN LOCALLY

No H200 training was performed in this local Windows session. Therefore there is no claim that RescueCredit improves policy ability. Run the cloud pilot and obey `outputs/pilot/gate.json` before any confirmatory claim.

## Readiness

- Engineering infrastructure: locally gated and ready for the cloud pilot; reviewer assurance remains same-family/provisional.
- Research evidence: insufficient until real model pilot and confirmatory runs finish.
- Ready for result-to-claim: NO.
