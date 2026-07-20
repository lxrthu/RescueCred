# ToolSandbox receipt-horizon statistical result

**Date:** 2026-07-21

**Remote artifact:** `outputs/toolsandbox_receipt_horizon_stats_seed42/statistical_audit.json`

**Status:** completed; positive routing claim not supported.

## Paired frozen-task results

| Metric | V7 one-step | V9 two-step | V9 - V7 | 95% task-bootstrap CI |
|---|---:|---:|---:|---:|
| Cross-task ROC-AUC | 0.595696 | 0.473745 | -0.121951 | [-0.227354, -0.019243] |
| Reverse recall | 0.129412 | 0.129412 | 0.000000 | [-0.050633, 0.045455] |
| Rescue drop | 0.000000 | 0.048780 | +0.048780 | [0.000000, 0.111111] |
| Probe rate | 0.230159 | 0.230159 | 0.000000 | [0.000000, 0.000000] |

The primary paired task-swap test classifies V9 as worse than V7: one-sided
`p=0.012249` for a lower V9 AUC and two-sided `p=0.023549`. Only 1.02% of the
20,000 task-bootstrap AUC deltas were above zero. Reverse recall did not improve;
the Rescue-drop increase is a harmful point estimate but is not significant in
the exploratory two-sided permutation test (`p=0.497925`).

## Supported claim

On these 126 frozen events from 38 ToolSandbox tasks, extending the tested
deployment-visible receipt representation from one step to two significantly
reduces cross-task AUC and does not improve Reverse recall. Static, one-step,
and two-step variants all fail the frozen positive routing gate.

## Unsupported claims

- The evidence does not show that all deployment-visible information is absent.
- It does not establish a universal impossibility result for every router.
- It does not include retraining, multi-seed, or broader population uncertainty.
- It does not provide formal Rescue-risk certification.

## Frozen decision

Stop receipt-horizon expansion on these events. Do not run three-step or full-
trajectory variants, and do not tune another loss/head on the same labels.
Future positive routing work must introduce qualitatively new visible evidence,
such as structured state diffs, explicit precondition predicates, or verifier
feedback, and evaluate it on fresh tasks.

## Evidence identity and caveat

- Task set SHA-256: `41db0e0587119d081a19340e664416b768330bc96d31bc5611fd30148d708e4b`
- V7 OOF SHA-256: `a4938ad77533a0e687a79d4cde52410d6af1ba3941c68bdf99a26537cc1643c8`
- V9 OOF SHA-256: `607191231f89b1c92df9aaebd136265b1ccab17b6ceb7da1d44b86ad92416b79`
- Protocol SHA-256: `46c2195b39ea2e1010b73eb4ba12f9058b1a91571966763a7c3af44858e72403`

The statistical result is conditional on frozen OOF predictions. The remote
JSON was supplied in the conversation; semantic review is same-family and
provisional.
