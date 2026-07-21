# Compensation Trap Experiment Tracker

| Run ID | Purpose | Split | Priority | Status | Gate |
|---|---|---|---|---|---|
| CT000 | inventory historical hashes | all frozen artifacts | MUST | READY_SERVER | dynamic protocol/audit inventory implemented |
| CT001 | seal untouched scenarios | ToolSandbox offset 205 tail | MUST | READY_SERVER | zero historical overlap + offset-tail sentinel |
| CT010 | exact public-signature collisions | frozen ToolSandbox | MUST | READY_SERVER | ≥5 opposing pairs / ≥3 tasks |
| CT011 | approximate cross-task collisions | frozen ToolSandbox | MUST | READY_SERVER | ≥20 label-blind MNN pairs at similarity ≥0.90 |
| CT020 | failure-family synthesis | frozen artifacts | MUST | TODO | exact metric recomputation |
| CT030 | benchmark package/verifier | frozen public/private banks | MUST | READY_SERVER | clean-room verify + tamper rejection; release blocked pending license review |
| CT040 | fresh Exact Shadow confirmation | sealed CT001 tasks | MUST | READY_SERVER | both directions, coverage, bootstrap |
| CT050 | strong public LLM judge | task-disjoint frozen bank | MUST | BLOCKED_BY_CT010 | AUC/calibration/swap consistency |
| CT060 | paper tables and figures | sealed CT outputs | MUST | BLOCKED | all numbers artifact-bound |
