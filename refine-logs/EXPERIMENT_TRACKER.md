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
| TS000 | ToolSandbox pinned contract probe | READY_CLOUD | `scripts/cloud/setup_toolsandbox_stage0.sh`; commit, snapshot, schema, evaluator gates |
| TS001 | ToolSandbox 3-scenario signal integration smoke | READY_CLOUD | automatic first stage of `scripts/cloud/run_toolsandbox_signal_audit.sh` |
| TS002 | ToolSandbox 40-scenario Harness/Shadow audit | READY_CLOUD | natural visible-error repair plus controlled missing-argument signal; no V3.1 training yet |
| TS003 | ToolSandbox implementation review | DEPLOY_YES | pinned-source review resolved scenario-count, role-boundary, gate-accounting, console-snapshot, and timeout blockers |
| TS004 | ToolSandbox terminal-credit H8 audit | NEGATIVE_SPARSE | 36 valid controlled; 1 nonzero (2.78%); natural 1 zero-delta; no training |
| TS005 | V4 lexicographic-regret fresh audit | READY_CLOUD_TIMEOUT_REPAIRED | sanity produced one progress rescue; 180s timeout cascade fixed with frozen 600s ceiling and stateless worker restart; full suite passes |
| TS006 | V4.1 Tool-ID Harness diagnostic and fresh audit | DONE_GATE_PASS | 40/40 schema-complete proposals; 33/33 controlled nonzero; natural 3 rescues, 0 harms; worker failures 2.5%; exact snapshots |
| TS007 | V4.1 same-data Mask vs causal preference seed 42 | DONE_NEGATIVE_DIRECTIONAL_ONLY | 33 held-out causal events; no selection flips or accuracy gain, but all 3 reverse margins moved in the correct direction (-0.0064, -0.0175, -0.0182) |
| TS008 | V4.2 balanced-margin Mask vs causal preference seed 42 | DONE_NEGATIVE_DIRECTIONAL_ONLY | 33 held-out events; no flips or accuracy gain; all 3 reverse margins moved correctly (-0.0443, -0.1145, -0.1289), but rescue events also drifted toward A; offset-165 remained untouched |
| TS009 | V4.3 multi-prefix anchored Mask vs causal preference seed 42 | DONE_NEGATIVE_DATA_GATE | 75 nonzero events across 33 tasks, but only 4 reverse events from 4 tasks versus frozen 8/5 gate; exact snapshots/privacy/diversity checks passed; stopped before training and preserved offset-165 |
| TS010 | V4.4 reference-free both-valid candidate-diversity audit | READY_CLOUD | Same offset-85 training tasks; proposal A plus up to 3 distinct public-schema-valid alternatives with visible-value provenance; max 4 pairs/task; sanity first, then >=60 nonzero and bidirectional 8-event/5-task gate; no adapter training yet |
