# DeltaGuard Experiment Tracker

| Run | Milestone | Status | Evidence / command |
|---|---|---|---|
| DG000 | compile | DONE | `python -m compileall -q ...` |
| DG001 | deterministic unit tests | DONE | 9 passed; full repository suite passed with 4 existing skips |
| DG002 | independent implementation review | DONE | DEPLOY YES; Public Paired Deltas sanity/feasibility scope |
| DG010 | ToolSandbox sanity | READY_SERVER | `run_toolsandbox_deltaguard_seed42.sh sanity ...` |
| DG020 | fixed feasibility | BLOCKED_BY_SANITY | sanity pass 后执行 |
| DG030 | 240-event full pilot | BLOCKED_BY_FEASIBILITY | 需要每类 80 个 label-blind source events |
| DG040 | independent Rescue certification | BLOCKED_BY_FULL_GATE | full gate 通过后单独冻结 |
