# Research Findings

## 2026-07-16 — AppWorld reference-free Harness gate

- **Tested**: Qwen and Azure GPT-4o candidate selection, including public OpenAPI constraints and visible candidate provenance.
- **Final evidence**: On the untouched 27-task dev holdout, the provenance-aware Azure Harness made 34 repairs, of which 23 were correct: precision 67.6%, coverage 29.1%, single-step rescue rate 19.7%, and observed clean-case harm 0%.
- **Verdict**: The frozen 90% precision gate failed. The Harness is not authorized as a teacher for RescueCredit-v2.
- **What failed**: Automatic reference-free correction generation did not reach reliable holdout precision even after adding provenance.
- **What remains reusable**: AppWorld rollback, public-schema matching, reference isolation, provenance extraction, audit logging, and the implemented V2 loss machinery.
- **Do not repeat**: Do not tune candidate-count, confidence, or prompt thresholds against the already observed train/dev partitions. Do not present Oracle/API-Bank corrections as a deployable reference-free Harness.
- **Pivot**: Treat correction generation as an external component. Evaluate the causal credit-assignment method using frozen verified correction records shared by all compared methods.
- **Evidence caveat**: The local artifact is a transparent transcription of user-pasted remote output; an independent integrity audit was unavailable.

## 2026-07-16 — Route-A deterministic immediate-effect diagnostic

- **Tested**: 55 frozen AppWorld dev events, replaying the same prefix and scoring A/B immediately with the official evaluator; no Azure, continuation, reference suffix, or test data.
- **Result**: 55 valid pairs but only one nonzero A/B effect. Mask and V2 both scored 0.25655844155844154 on average, both had causal selection accuracy 0, and all 55 method outcomes tied.
- **Verdict**: The claim that RescueCredit-v2 selects causally better corrections than Mask is not supported. The preregistered gate failed.
- **What failed**: The event/evaluator combination was almost entirely causally uninformative, and V2 did not choose the beneficial action in the only informative event.
- **Do not repeat**: Do not run more seeds under this unchanged immediate-scoring protocol and do not promote the earlier 4/7 preference result or +0.001039 continuation delta as main evidence.
- **Next route**: Build a preregistered set of causally informative disagreement events or a bounded-horizon evaluator that captures delayed effects without reference leakage, then repeat paired interventions.
- **Evidence caveat**: Remote JSON was user-pasted and locally transcribed; semantic review is same-family and provisional.

## 2026-07-21 — ToolSandbox receipt-horizon observability boundary

- **Tested**: V7 one-step versus V9 two-step deployment-visible A/B receipt representations on the same 126 OOF events from 38 frozen ToolSandbox tasks, using 20,000 whole-task bootstrap samples and 20,000 paired task-swap permutations.
- **Primary result**: Cross-task ROC-AUC decreased from 0.595696 to 0.473745 (delta -0.121951; 95% task-bootstrap CI [-0.227354, -0.019243]; one-sided paired permutation p=0.012249; two-sided p=0.023549).
- **Operating point**: Reverse recall remained 0.129412, probe rate remained 0.230159, and empirical Rescue drop increased from 0 to 0.048780. The Rescue-drop comparison is exploratory and not statistically significant.
- **Verdict**: The positive conservative-routing claim is not supported. Extending the tested receipt representation from one to two steps significantly worsens paired frozen-task AUC and does not improve Reverse recall.
- **Supported boundary**: Exact Shadow replay can recover ex-post Rescue/Reverse labels, but the tested static, one-step, and two-step representations do not make those labels reliably actionable cross-task under the frozen 2% Rescue-harm gate.
- **Do not repeat**: Do not try three-step/full-trajectory receipt features or further loss/head tuning on these same 126 events. Do not generalize this result into a universal information-impossibility claim.
- **Future positive route**: Introduce qualitatively new deployment-visible information—structured state diffs, explicit precondition predicates, or verifier feedback—and evaluate on fresh tasks.
- **Evidence caveat**: Inference is conditional on frozen OOF predictions and excludes retraining and population uncertainty. Remote JSON was user-pasted; same-family review remains provisional.
