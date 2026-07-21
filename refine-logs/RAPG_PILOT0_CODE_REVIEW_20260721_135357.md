# RAPG Surrogate Preflight Code Review

## Verdict

**DEPLOY YES — candidate-selector surrogate preflight only.**

## Verified

- Locked `audit_rate=0.20` and `p_min=0.05` propensities are independently rebuilt in the gate.
- Oracle, HT estimates, exact full-LoRA-gradient design MSE, gains, task effects, cosine, ESS, weights, and expected cost are recomputed.
- Actual behavior and propensity ledgers and all source/bank/prediction/estimate hashes are checked.
- All replay-valid both-valid pairs are exported without Rescue/Reverse direction filtering.
- Public context, executed B return, and private Shadow-A return are physically separated; the score builder never opens Shadow-A.
- Full Shadow ground truth comes from the official ToolSandbox evaluator.
- A pass authorizes only clean on-policy autoregressive collection, never a policy/RAPG positive claim.

## Non-blocking cautions

- If sampled replacements are fewer than `0.20*N`, evaluation exits as a feasibility failure before writing a gate file.
- Server sanity must report `4 passed` and no skipped Torch tests.
- Qwen2.5-7B LoRA is feasible on H200; a 24GB GPU may OOM because both candidate graphs coexist.
