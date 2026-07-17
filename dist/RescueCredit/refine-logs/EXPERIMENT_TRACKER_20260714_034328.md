# Experiment Tracker

| Run | Milestone | Status | Evidence / command |
|---|---|---|---|
| R000 | compile | DONE | `python -m compileall -q ...` |
| R001 | unit tests | DONE | 23 passed |
| R002 | API-Bank prepare | DONE | 254 executable; 45 mixed-tool conflicts excluded; frozen 138/19/23/29 splits |
| R003 | Toy MDP | DONE | exact Q + estimator JSON/CSV |
| R004 | API-Bank injected smoke | DONE | 114 episodes; replay failure 0 |
| R100 | seed-42 pilot | PENDING_CLOUD | `scripts/cloud/run_pilot_4gpu.sh` |
| R200-R208 | confirmatory | BLOCKED_BY_PILOT_GATE | `scripts/cloud/run_confirmatory_4gpu.sh` |
