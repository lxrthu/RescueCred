# ToolSandbox V4.1 Same-Data Preference Comparison

Status: frozen design before training or offset-125 evaluation outcomes
Date: 2026-07-20

## Question

Does V4 lexicographic counterfactual credit teach a policy to select the
causally better correction branch more reliably than blanket Mask/Correction
training when both methods receive exactly the same event presentations?

## Frozen training source

- The passed V4.1 Tool-ID fresh audit at seed 42, offset 85, H8.
- Include replay-valid nonzero controlled and natural repair pairs only.
- Reconstruct prompts from initial deployment-visible history, the relevant
  public tool schema, and candidate actions A/B.
- Preference labels and official outcomes never enter the prompt.
- Mask always teaches B over A. V4 follows the frozen lexicographic direction.
- Both methods use unit preference weight, identical natural event order,
  identical presentations, optimizer, LoRA shape, epochs, and base model. This
  isolates causal direction rather than weight scaling or resampling.

## Frozen evaluation

- ToolSandbox seed 42, offset 125, limit 40, H8, search depth 8.
- Scenario hashes are frozen before either adapter is trained.
- Exclude all old V4, V4.1 diagnostic, and V4.1 training-audit scenarios.
- The Tool-ID Harness, DeepSeek continuation policy, V4 credit, official
  evaluator, and snapshot rules are unchanged.
- Models score only public prompts and candidates. Official branch outcomes are
  joined after scoring.

## Gate

All integrity checks must pass. The outcome gate requires at least 20 valid
nonzero evaluation pairs, at least two reverse-preference pairs, at least three
selection disagreements, V4 causal-selection accuracy at least five percentage
points above Mask, more V4 wins than losses on disagreements, and no decrease
in mean selected terminal similarity or bounded progress AUC.

This is a controlled-state preference-learning result, not autonomous
ToolSandbox task success. A pass authorizes multi-seed confirmation and then a
fully autonomous evaluation; a failure stops expansion without threshold
tuning.
