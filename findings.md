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
