# ToolSandbox V5 local-only code review

## Verdict

`DEPLOY_SANITY_YES`. Run the frozen server script, which performs tests and
feature extraction before training either router. This review does not
authorize a paper or fresh-confirmation claim.

## Method correctness

- The Qwen base model and exact V4.5 Mask adapter remain frozen.
- Training labels are KEEP/FLIP relative to the frozen Mask selection, rather
  than new generative preference gradients.
- The semantic router uses a deterministic projection of the frozen
  completion-representation difference plus Mask margin features.
- A margin-only router is trained on the same events as a control.
- Thresholds are selected by deterministic task-grouped out-of-fold training
  predictions; development and post-hoc outcomes never select the threshold.
- The public scorer has no private-outcome argument. Official outcomes are
  joined only by a separate evaluation script after prediction files freeze.

## Identity and leakage checks

- Protocol binds V4.4 training data, frozen Mask/base identities, the negative
  V4.6 result, source files, and known development/post-hoc split manifests.
- Feature cache, router checkpoints, public predictions, result rows, and raw
  metric recomputation are hash-bound by the development gate.
- Task hashes define OOF folds, preventing candidate pairs from one task from
  appearing in both router-training and OOF-validation partitions.

## Validation

- Ruff: passed.
- Python compilation: passed.
- Bash syntax: passed.
- Directed V5/V4.6/V4.5/preference tests: passed; the torch-only head test is
  skipped locally because the Windows test environment lacks torch and will run
  in the server preflight.
- Full repository suite: passed with the two existing skips plus the local-only
  torch skip.
- `git diff --check`: passed.

## Remaining deployment risk

The first server stage must validate direct access to the Qwen causal-LM
backbone and PyTorch safe checkpoint loading in the actual GPU environment.
The run stops before evaluation if feature extraction or task-grouped OOF
training fails. Even a development pass requires a newly frozen ToolSandbox
scenario profile before confirmation.
