# Experiment Code Review

## Assurance status

`provisional / same-family review`

Two secondary-agent review passes were recorded under `.aris/traces/experiment-bridge/2026-07-14_run01/`. The second review still blocked H200 execution on five concrete issues. Those issues were subsequently fixed and verified locally; no claim of independent external acceptance is made.

## Post-review fixes

- Schema-valid off-path tool calls now consume a step and return `no_effect`, allowing delayed recovery instead of immediate termination.
- Deterministic environment support produces both 0 and 1 counterfactual returns (`Var(G0)=0.25` on dev); real model Full Shadow separately hard-fails unless `Var(G0)>0`.
- Shadow history records actual tool receipts; horizon-censored replay is `replay_valid=false`.
- LoRA dropout is zero and rollout/scoring runs in eval mode, preventing behavior-ratio dropout noise.
- Content-addressed snapshots persist task, environment state and RNG; an event JSONL can restore the snapshot in a fresh environment and verify `state_hash`.
- Aggregation rejects duplicate task IDs, verifies eval/run method and seed plus cross-method split hashes and full training comparability, averages seeds within task, then bootstraps tasks.

## Local gates

- 26 tests passed.
- Ruff passed on project-owned code; the frozen upstream API-Bank source is excluded.
- `compileall` passed.
- All cloud shell scripts passed `bash -n`.
- API-Bank controlled split/leakage gate passed and the dry-run completed.

## Remaining research gate

The local machine did not run Qwen training or Full Shadow on H200. The cloud pilot must still pass `outputs/pilot/gate.json`, including the model-derived `Var(G0)>0` identifiability gate, before confirmatory runs are allowed.
