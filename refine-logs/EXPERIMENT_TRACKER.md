# Experiment Tracker

| Run | Milestone | Status | Evidence / command |
|---|---|---|---|
| R000 | compile | DONE | `python -m compileall -q ...` |
| R001 | unit tests | DONE | 26 passed |
| R002 | API-Bank prepare | DONE | 254 executable; 45 mixed-tool conflicts excluded; frozen 138/19/23/29 splits |
| R003 | Toy MDP | DONE | exact Q + estimator JSON/CSV |
| R004 | API-Bank injected smoke | DONE | 114 episodes; replay failure 0 |
| R005 | delayed-recovery G0 support | DONE | 10 zero + 10 success returns; variance 0.25 |
| R100 | seed-42 pilot | PENDING_CLOUD | `scripts/cloud/run_pilot_4gpu.sh` |
| R200-R208 | confirmatory | BLOCKED_BY_PILOT_GATE | `scripts/cloud/run_confirmatory_4gpu.sh` |
| R100-v2-full | deployable full-data seed-42 pilot | NEGATIVE_DEGENERATE | Mask-v2 = RescueCredit-v2 = 7/18; zero eligible audits |
| R101-v2-curriculum | shared visible-structure sampling sanity | READY_CLOUD | `scripts/cloud/run_v2_visible_curriculum_smoke_2gpu.sh`; gate: 5 audits and 3 nonzero causal events |
