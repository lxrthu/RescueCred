# Research Output Manifest

> Auto-maintained by ARIS skills. Tracks all generated artifacts across the research lifecycle.

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-07-14 03:43 | /experiment-bridge | idea-stage/docs/research_contract.md | implementation | focused RescueCredit claims and gates |
| 2026-07-14 03:43 | /experiment-bridge | refine-logs/EXPERIMENT_PLAN_20260714_034328.md | implementation | timestamped experiment plan |
| 2026-07-14 03:43 | /experiment-bridge | refine-logs/EXPERIMENT_PLAN.md | implementation | latest experiment plan |
| 2026-07-14 03:43 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER_20260714_034328.md | implementation | timestamped run tracker |
| 2026-07-14 03:43 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest run tracker |
| 2026-07-14 03:43 | /experiment-bridge | refine-logs/EXPERIMENT_RESULTS_20260714_034328.md | implementation | timestamped initial results |
| 2026-07-14 03:43 | /experiment-bridge | refine-logs/EXPERIMENT_RESULTS.md | implementation | latest initial results |
| 2026-07-14 03:43 | /experiment-bridge | data/api_bank_controlled_v1/manifest.json | implementation | frozen official-source task counts and split hashes |
| 2026-07-14 03:43 | /experiment-bridge | outputs/toy/summary.json | implementation | exact MDP and estimator sanity output |
| 2026-07-14 03:43 | /experiment-bridge | outputs/smoke/api_bank/summary.json | implementation | non-evidence injected infrastructure smoke |
| 2026-07-14 03:43 | /experiment-bridge | docs/CLOUD_RUN_CN.md | implementation | H200 execution and API key guide |
| 2026-07-15 15:55 | /experiment-bridge | refine-logs/DEPLOYABLE_HARNESS_CODE_REVIEW_20260715_155529.md | implementation | timestamped local-only Stage 1 code review |
| 2026-07-15 15:55 | /experiment-bridge | refine-logs/DEPLOYABLE_HARNESS_CODE_REVIEW.md | implementation | latest deployable Harness review and gate verdict |
| 2026-07-15 15:55 | /experiment-bridge | docs/DEPLOYABLE_HARNESS_STAGE1_CN.md | implementation | reference isolation, quality thresholds and current audit result |
| 2026-07-15 15:55 | /experiment-bridge | outputs/deployable_harness_stage1_patch.zip | implementation | server-ready Stage 1 source patch bundle |
| 2026-07-15 15:55 | /experiment-bridge | environments/api_bank/correction_generator.py | implementation | frozen reference-free missing-argument proposal generator |
| 2026-07-15 15:55 | /experiment-bridge | scripts/cloud/run_deployable_harness_qwen_gate.sh | implementation | single-GPU frozen-Qwen Harness quality gate |
| 2026-07-15 15:55 | /experiment-bridge | outputs/PASTE_ON_SERVER_DEPLOYABLE_HARNESS.txt | implementation | paste-only server installer with backup, tests and GPU gate |
| 2026-07-15 16:27 | /experiment-bridge | refine-logs/RESCUECREDIT_V2_CODE_REVIEW_20260715_162705.md | implementation | timestamped local-only V2 code review |
| 2026-07-15 16:27 | /experiment-bridge | refine-logs/RESCUECREDIT_V2_CODE_REVIEW.md | implementation | latest V2 implementation review and remaining GPU gate |
| 2026-07-15 16:27 | /experiment-bridge | docs/RESCUECREDIT_V2_IMPLEMENTATION_CN.md | implementation | V2 objective, budgets, logs and smoke instructions |
| 2026-07-15 16:27 | /experiment-bridge | scripts/cloud/run_v2_smoke_2gpu.sh | implementation | two-GPU V2 causal-loss smoke gate |
| 2026-07-15 16:27 | /experiment-bridge | outputs/rescuecredit_v2_patch.zip | implementation | server-ready V2 source patch bundle |
| 2026-07-15 16:27 | /experiment-bridge | outputs/PASTE_ON_SERVER_RESCUECREDIT_V2.txt | implementation | paste-only V2 installer with backup, focused tests and dry-run |
| 2026-07-16 03:59 | /paper-plan | PAPER_PLAN_20260716_035646.md | paper | timestamped AAAI conditional-credit paper plan |
| 2026-07-16 03:59 | /paper-plan | PAPER_PLAN.md | paper | latest AAAI conditional-credit paper plan |
# Route-A frozen correction bank (2026-07-16)

- `rescuecredit/frozen_bank.py`: immutable bank schema, hashing, and leakage guards.
- `scripts/build_appworld_route_a_bank.py`: train-only AppWorld public/private bank builder.
- `scripts/check_route_a_bank.py`: pre-training integrity gate.
- `scripts/cloud/run_appworld_route_a_bank.sh`: cloud entrypoint including AppWorld environment setup.
- `docs/ROUTE_A_FROZEN_BANK_CN.md`: Chinese server run guide.
- `dist/PASTE_ROUTE_A_BANK_TO_SERVER.sh`: password-free paste bundle generated locally.

## Route-A Shadow credit smoke

- `rescuecredit/appworld_shadow_credit.py`: official-score extraction and causal decision logic.
- `scripts/appworld_azure_continuation_worker.py`: persistent reference-free Azure continuation worker.
- `scripts/attach_appworld_shadow_credit.py`: same-state A/B AppWorld branch evaluator.
- `scripts/check_route_a_shadow_gate.py`: nonzero causal-support gate before training.
- `scripts/cloud/run_route_a_shadow_smoke.sh`: 20-event cloud smoke entrypoint.
- `dist/PASTE_ROUTE_A_SHADOW_TO_SERVER.sh`: password-free paste-and-launch bundle.

## Route-A dense official requirement credit

- `scripts/recompute_route_a_dense_credit.py`: reuses saved A/B evaluator reports; no new model calls.
- `scripts/check_route_a_dense_gate.py`: requires usable positive and negative causal support.
- `scripts/cloud/run_route_a_dense_recompute.sh`: offline recomputation entrypoint.
- `dist/PASTE_ROUTE_A_DENSE_TO_SERVER.sh`: password-free install-and-run bundle.

## Route-A same-bank Mask vs RescueCredit-v2 preference pilot

- `rescuecredit/route_a_preference.py`: deterministic split and method-specific preference routing.
- `scripts/prepare_route_a_preference_data.py`: freezes one shared train/validation partition.
- `scripts/train_route_a_preference.py`: length-normalized LoRA DPO-style preference learner.
- `scripts/evaluate_route_a_preference.py`: held-out causal-direction accuracy diagnostic.
- `scripts/check_route_a_preference_gate.py`: engineering gate before AppWorld task evaluation.
- `scripts/cloud/run_route_a_seed42_preference_pair.sh`: automatic two-GPU parallel runner.
- `docs/ROUTE_A_SEED42_PREFERENCE_PILOT_CN.md`: Chinese cloud run and result guide.
- `dist/PASTE_ROUTE_A_SEED42_TO_SERVER.sh`: password-free paste-and-launch bundle.

## Route-A AppWorld dev paired task-score evaluation

- `rescuecredit/route_a_task_eval.py`: reference-free action validation, summaries, and paired gate.
- `scripts/build_route_a_appworld_dev_events.py`: freezes unseen dev controlled-state events.
- `scripts/route_a_adapter_worker.py`: persistent local LoRA adapter inference worker.
- `scripts/evaluate_route_a_appworld_dev.py`: official AppWorld requirement-score evaluation.
- `scripts/check_route_a_appworld_dev_gate.py`: pre-registered paired task-score gate.
- `scripts/cloud/run_route_a_appworld_dev_pair.sh`: GPU sanity plus two-GPU paired dev runner.
- `docs/ROUTE_A_APPWORLD_DEV_EVAL_CN.md`: Chinese evaluation contract and commands.
- `dist/PASTE_ROUTE_A_APPWORLD_DEV_TO_SERVER.sh`: password-free paste-and-launch bundle.

## ToolSandbox Harness/Shadow signal audit

- `environments/toolsandbox/adapter.py`: pinned official ToolSandbox snapshot, tool execution, visible-history, and evaluator adapter.
- `rescuecredit/toolsandbox_audit.py`: natural/controlled signal summaries and frozen promotion gate.
- `scripts/inspect_toolsandbox_contract.py`: exact commit, snapshot restore, schema, and evaluator contract probe.
- `scripts/toolsandbox_azure_worker.py`: reference-free proposal, repair, and continuation worker.
- `scripts/audit_toolsandbox_signal.py`: paired branch audit using official `EvaluationResult.similarity`.
- `scripts/cloud/setup_toolsandbox_stage0.sh`: isolated Python 3.9 environment and pinned official checkout.
- `scripts/cloud/run_toolsandbox_signal_audit.sh`: 3-scenario smoke followed by 40-scenario audit.
- `tests/test_toolsandbox_audit.py`: corruption, worker validation, three-way credit, and gate tests.
- `docs/TOOLSANDBOX_SIGNAL_AUDIT_CN.md`: cloud commands, output contract, and interpretation limits.

## ToolSandbox V4.1 same-data preference comparison

- `rescuecredit/toolsandbox_preference.py`: public prompt, matched ordering, causal routing, and independent metric recomputation.
- `scripts/prepare_toolsandbox_v41_preference_data.py`: separates public model inputs from private official outcomes.
- `scripts/train_toolsandbox_v41_preference.py`: same-event/same-budget Mask versus V4 LoRA preference training.
- `scripts/evaluate_toolsandbox_v41_preference.py`: fresh candidate scoring with post-scoring official-outcome joins.
- `scripts/freeze_toolsandbox_v41_preference_protocol.py`: binds training data, untouched evaluation scenarios, model, sources, and thresholds before outcomes.
- `scripts/check_toolsandbox_v41_preference_gate.py`: independently recomputed integrity and outcome gate.
- `scripts/cloud/run_toolsandbox_v41_preference_seed42.sh`: two-GPU seed-42 training plus offset-125 fresh evaluation runner.
- `tests/test_toolsandbox_preference.py`: direction, privacy boundary, matched-budget, metric, gate, and runner-order tests.
- `refine-logs/TOOLSANDBOX_V41_PREFERENCE_PLAN_20260720.md`: timestamped preregistration record.
- `refine-logs/TOOLSANDBOX_V41_PREFERENCE_PLAN.md`: latest plan pointer.
- `refine-logs/TOOLSANDBOX_V41_PREFERENCE_CODE_REVIEW_20260720.md`: timestamped local-only pre-deploy review.
- `refine-logs/TOOLSANDBOX_V41_PREFERENCE_CODE_REVIEW.md`: latest review pointer.

## ToolSandbox V4.2 balanced-margin comparison

- `scripts/train_toolsandbox_v42_preference.py`: identical class-balanced event sequence and absolute-margin objective for Mask and V4.2.
- `scripts/evaluate_toolsandbox_v42_preference.py`: public-only candidate scoring for development and confirmation roles.
- `scripts/freeze_toolsandbox_v42_protocol.py`: binds the offset-85 training source, known offset-125 development artifacts, untouched offset-165 identity, model, source, sequence, and thresholds before training.
- `scripts/check_toolsandbox_v42_gate.py`: independently recomputed development/confirmation integrity and outcome gates.
- `scripts/cloud/run_toolsandbox_v42_seed42.sh`: two-GPU training, zero-API development gate, then conditionally authorized offset-165 confirmation.
- `tests/test_toolsandbox_v42.py`: balance, direction, objective, gate, and execution-order regression tests.
- `refine-logs/TOOLSANDBOX_V42_PLAN_20260720.md`: timestamped frozen V4.2 design.
- `refine-logs/TOOLSANDBOX_V42_PLAN.md`: latest V4.2 plan pointer.
- `refine-logs/TOOLSANDBOX_V42_CODE_REVIEW_20260720.md`: timestamped local-only deployment review.
- `refine-logs/TOOLSANDBOX_V42_CODE_REVIEW.md`: latest V4.2 review pointer.
| 2026-07-16 19:11 | /experiment-bridge | refine-logs/ROUTE_A_APPWORLD_DEV_CODE_REVIEW_20260716_191149.md | implementation | timestamped local-only dev evaluation review |
| 2026-07-16 19:11 | /experiment-bridge | refine-logs/ROUTE_A_APPWORLD_DEV_CODE_REVIEW.md | implementation | latest dev evaluation review pointer |
| 2026-07-16 19:11 | /experiment-bridge | refine-logs/ROUTE_A_APPWORLD_DEV_TRACKER_20260716_191149.md | implementation | timestamped sanity and full dev run tracker |
| 2026-07-16 19:11 | /experiment-bridge | refine-logs/ROUTE_A_APPWORLD_DEV_TRACKER.md | implementation | latest dev evaluation tracker pointer |
| 2026-07-16 19:13 | /experiment-bridge | dist/PASTE_ROUTE_A_APPWORLD_DEV_TO_SERVER.sh | implementation | paste-only server installer and paired dev launcher |
| 2026-07-16 19:45 | /experiment-bridge | refine-logs/ROUTE_A_APPWORLD_DEV_CODE_REVIEW_20260716_194520.md | implementation | V2 review after identifying and removing the reference-suffix ceiling |
| 2026-07-16 19:45 | /experiment-bridge | refine-logs/ROUTE_A_APPWORLD_DEV_TRACKER_20260716_194520.md | implementation | V1 ceiling verdict and V2 pending execution tracker |
| 2026-07-16 20:34 | /experiment-bridge | refine-logs/ROUTE_A_BOUNDED_PLAN_20260716_203410.md | implementation | preregistered seed-42 AppWorld 4/8-step bounded-horizon protocol |
| 2026-07-16 20:34 | /experiment-bridge | refine-logs/ROUTE_A_BOUNDED_PLAN.md | implementation | latest bounded-horizon plan pointer |
| 2026-07-16 20:54 | /experiment-bridge | rescuecredit/route_a_bounded.py | implementation | bounded-horizon summaries, exact protocol constants, and strict gate |
| 2026-07-16 20:54 | /experiment-bridge | scripts/freeze_route_a_bounded_protocol.py | implementation | pre-outcome event and selection hash lock |
| 2026-07-16 20:54 | /experiment-bridge | scripts/evaluate_route_a_bounded.py | implementation | cached visible-only H4/H8 A/B evaluator with prefix verification |
| 2026-07-16 20:54 | /experiment-bridge | scripts/check_route_a_bounded_gate.py | implementation | non-overrideable primary H8 gate entrypoint |
| 2026-07-16 20:54 | /experiment-bridge | scripts/cloud/run_route_a_appworld_bounded.sh | implementation | remote sanity-first tmux runner |
| 2026-07-16 20:54 | /experiment-bridge | tests/test_route_a_bounded.py | implementation | bounded metric, cache, prefix, and gate tests |
| 2026-07-16 20:54 | /experiment-bridge | tests/test_route_a_bounded_contract.py | implementation | static deployment-contract regression tests |
| 2026-07-16 20:54 | /experiment-bridge | docs/ROUTE_A_APPWORLD_BOUNDED_CN.md | implementation | Chinese server run and return guide |
| 2026-07-16 20:54 | /experiment-bridge | refine-logs/ROUTE_A_BOUNDED_CODE_REVIEW_20260716_205435.md | implementation | three-round same-family review ending DEPLOY YES |
| 2026-07-16 20:54 | /experiment-bridge | refine-logs/ROUTE_A_BOUNDED_CODE_REVIEW.md | implementation | latest bounded review pointer |
| 2026-07-16 20:54 | /experiment-bridge | refine-logs/ROUTE_A_BOUNDED_TRACKER_20260716_203410.md | implementation | remote-run tracker and validation status |
| 2026-07-16 20:54 | /experiment-bridge | refine-logs/ROUTE_A_BOUNDED_TRACKER.md | implementation | latest bounded tracker pointer |
| 2026-07-16 20:54 | /experiment-bridge | dist/PASTE_ROUTE_A_BOUNDED_TO_SERVER.sh | implementation | password-free install, test, and launch bundle |
| 2026-07-18 23:55 | /experiment-bridge | refine-logs/TOOLSANDBOX_V4_POOL_ERRATUM_20260718.md | preflight repair | documents the pre-outcome tiered scenario-pool correction |
| 2026-07-18 23:58 | /experiment-bridge | refine-logs/TOOLSANDBOX_V4_POOL_CODE_REVIEW_20260718.md | review | timestamped local-only pre-deploy review of the pool repair |
| 2026-07-18 23:58 | /experiment-bridge | refine-logs/TOOLSANDBOX_V4_POOL_CODE_REVIEW.md | review | latest ToolSandbox V4 pool-repair review pointer |
| 2026-07-19 00:20 | /monitor-experiment | refine-logs/TOOLSANDBOX_V4_WORKER_TIMEOUT_ERRATUM_20260718.md | sanity repair | documents the pre-holdout timeout cascade and frozen correction |
| 2026-07-19 00:20 | /monitor-experiment | refine-logs/TOOLSANDBOX_V4_WORKER_CODE_REVIEW_20260718.md | review | local-only pre-deploy review of worker restart and timeout binding |
| 2026-07-19 10:00 | /experiment-bridge | refine-logs/TOOLSANDBOX_V41_PLAN_20260719.md | plan | timestamped pre-outcome Tool-ID Harness diagnostic and confirmation protocol |
| 2026-07-19 10:00 | /experiment-bridge | refine-logs/TOOLSANDBOX_V41_PLAN.md | plan | latest ToolSandbox V4.1 Tool-ID Harness plan pointer |
| 2026-07-19 10:30 | /experiment-bridge | refine-logs/TOOLSANDBOX_V41_CODE_REVIEW_20260719.md | review | timestamped local-only Tool-ID Harness deployment review |
| 2026-07-19 10:30 | /experiment-bridge | refine-logs/TOOLSANDBOX_V41_CODE_REVIEW.md | review | latest ToolSandbox V4.1 review pointer |
| 2026-07-20 12:00 | /experiment-bridge | refine-logs/TOOLSANDBOX_V41_PREFERENCE_PLAN_20260720.md | plan | timestamped same-data Mask versus V4 preference protocol |
| 2026-07-20 12:00 | /experiment-bridge | refine-logs/TOOLSANDBOX_V41_PREFERENCE_PLAN.md | plan | latest ToolSandbox V4.1 preference comparison plan |
| 2026-07-20 12:30 | /experiment-bridge | refine-logs/TOOLSANDBOX_V41_PREFERENCE_CODE_REVIEW_20260720.md | review | local-only review ending seed-42 deploy yes |
| 2026-07-20 12:30 | /experiment-bridge | refine-logs/TOOLSANDBOX_V41_PREFERENCE_CODE_REVIEW.md | review | latest ToolSandbox V4.1 preference review pointer |
