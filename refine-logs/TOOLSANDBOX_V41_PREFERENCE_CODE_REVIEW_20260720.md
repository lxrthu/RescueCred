# ToolSandbox V4.1 Preference Code Review

Status: local-only review (delegation unavailable under the active execution policy)
Date: 2026-07-20
Verdict: **DEPLOY YES for the frozen seed-42 engineering gate only**

## Contract reviewed

- Same passed offset-85 V4.1 event source for Mask and V4.
- Same event sequence, presentation count, optimizer, LoRA config, and base model.
- Mask always labels B over A; V4 changes only the direction on frozen reverse events.
- Offset-125 exact 40-scenario evaluation is frozen before training.
- Evaluation uses official ToolSandbox outcomes joined only after public candidate scoring.

## Blocking issues found and resolved

1. The first gate trusted JSON summaries. It now independently recomputes all
   selection accuracies and selected official terminal/progress means from the
   raw per-event result files.
2. The first training file carried official branch metrics that the trainer did
   not use. Those metrics are now absent from `train.jsonl` and remain only in
   the private evaluation artifact; training receives public prompts, actions,
   and frozen preference-label fields.
3. The evaluation lock originally allowed fewer than the planned 40 scenarios.
   It now requires exactly 40 at offset 125 and rejects any overlap with all
   prior V4/V4.1 partitions.
4. The training protocol source inventory now binds direct shared dependencies,
   including the base DPO log-probability implementation and ToolSandbox credit
   validation.

## Integrity checklist

- Dataset ground truth: official ToolSandbox `EvaluationResult.similarity` and
  its validated H8 trace; never another model output.
- Leakage: no evaluator values, decisions, milestones, minefields, reference
  actions, or suffix trajectories enter model prompts.
- Reproducibility: fixed seed/config, exact hashes, exact event sequence, source
  identity, base-model identity, and non-reusable output root.
- Comparison: both adapters receive identical event presentations and unit
  weights, isolating causal preference direction.
- Evaluation: fresh scenario hashes are frozen before adapters train; raw rows
  are independently recomputed by the final gate.
- Failure handling: sanity tests precede GPU work; training/evaluation child
  statuses are checked; a fresh audit must pass its mechanism gate.

## Non-blocking limitations

- Prompt reconstruction uses the initial deployment-visible history plus the
  relevant public schema and the two treatment-point candidates, not the full
  pre-treatment receipt prefix. This is frozen and shared across methods.
- The primary gate conditions on replay-valid nonzero causal pairs. It measures
  candidate preference learning, not population task success.
- A seed-42 pass authorizes multi-seed confirmation and later autonomous task
  evaluation; it is not itself a paper-level autonomous success result.

## Validation

- Directed ToolSandbox preference/V4/V4.1 tests: passed.
- Full repository pytest suite: passed with two existing skips.
- Ruff on all touched Python files: passed.
- Python compilation and Bash syntax: passed.
