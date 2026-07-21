# RAPG Surrogate Preflight Tracker

| Run | Milestone | Status | Evidence |
|---|---|---|---|
| RAPG-S0 | local compile/lint/tests | PASSED_LOCAL_LIMITED | ruff passed; 10 passed, 2 torch tests skipped because local Torch absent |
| RAPG-S1 | all-valid physical source split | READY_SERVER | `prepare_toolsandbox_rapg_preflight_data.py` |
| RAPG-S2 | candidate-policy LoRA score/norm extraction | READY_SERVER | `build_toolsandbox_rapg_bank.py` |
| RAPG-S3 | cross-fit allocation + 1,000 audit draws | BLOCKED_BY_S2 | `evaluate_toolsandbox_rapg_pilot0.py` |
| RAPG-S4 | independently recomputed gate | BLOCKED_BY_S3 | `check_toolsandbox_rapg_gate.py` |

Server runner requires Torch tests to execute without skips. Positive paper claim and policy training remain unauthorized.
