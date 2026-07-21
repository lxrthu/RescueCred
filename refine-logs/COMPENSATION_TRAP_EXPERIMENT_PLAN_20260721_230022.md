# Compensation Trap Paper Evidence Plan

**Problem**：Harness 的聚合成功信号同时包含 Rescue（B 优于 A）与 Reverse（A 优于 B），因而不能直接衡量底层 policy competence。  
**Method thesis**：Exact Shadow 提供可审计的事后配对标签；论文贡献是揭示“事后可识别”与“部署时可行动”之间的经验分离，而不是提出新的 RL loss。  
**Date**：2026-07-21 23:00

## Claim Map

| Claim | Minimum convincing evidence | Blocks |
|---|---|---|
| C1：Compensation Trap 在未触碰任务上复现 | 预先封存、与全部旧 task/scenario hash 不相交的任务；Exact Shadow 同时观察到 Rescue 与 Reverse；按任务报告区间和 constant-router oracle headroom | B1, B2 |
| C2：事后标签可恢复，但在已测试的部署可见表示下不可稳定行动 | 同一公开表示等价类或高相似邻域中存在相反标签；强公开信息基线保持 task-disjoint；所有结论限定到测试表示 | B3, B4 |
| Anti-claim：失败只是类别不平衡或普遍有害的 B-credit gradient | 分层/平衡指标、collision lower bound，以及 Naive/Firewall/IPW/AIPW 对 Full-Shadow oracle 的梯度结果 | B3, B5 |

明确不声称：信息在客观上不存在、所有路由/RL 方法都失败、Exact Shadow 改善 policy、或获得 SOTA。

## Paper Storyline

主文只保留：Compensation Trap 定义、Exact Shadow 审计、fresh confirmation、collision evidence、三类失败边界摘要。V6--V9 的逐版本过程移到附录，只按 `routing / probing / credit correction` 三类报告。

## Experiment Blocks

### B1：Untouched-task seal

- Claim：fresh confirmation 没有被旧结果或方法选择污染。
- Dataset：ToolSandbox 排序中尚未消费的最终 offset 205 后 13 个 scenarios；若清点发现任何 hash 已出现，整批停止，不顺延挑选。
- Setup：在打开任何 A/B outcome 前冻结 scenario hashes、模型、Harness、候选生成、continuation、H8 scorer、超时和 missingness rule。
- Integrity gate：与所有旧 protocol locks、candidate banks、development/confirmation artifacts 的 task/scenario hashes 零交集；恰好绑定全部可用 untouched scenarios。
- Target：Protocol/Appendix；MUST-RUN。

### B2：Fresh Compensation Trap confirmation

- Claim：未触碰任务中同时存在方向相反的 Harness effects。
- Compared systems：always-A、always-B、oracle per-event router；不训练新 learner。
- Metrics：valid paired events、Rescue/Reverse/Zero counts、涉及的独立 tasks、best-constant accuracy、oracle headroom、task-cluster bootstrap intervals、branch failure/missingness。
- Success criterion：至少 20 个 valid nonzero events；Rescue 与 Reverse 各至少 3 个且各覆盖至少 3 个 tasks；oracle 相对 best constant 的 headroom > 0，95% task-bootstrap 中观察到两方向的概率均 ≥ 0.95。
- Failure：只报告未复现，不扩大或更换 fresh set。
- Target：Table 1 主结果；MUST-RUN。

### B3：Public-state collision audit

- Claim：至少在预注册的部署可见表示下，相似输入可对应相反标签。
- Exact signature：公开 visible history 中的 tool sequence、A/B tool identity、参数 present/missing/type pattern、相关公开 schema shape/relation；不使用内部 scenario name、内部 prefix index、evaluator、reference、A/B outcome 或 hidden state。
- Approximate signature：同一公开字段的固定 hashing-TFIDF/character n-gram；阈值和 tie rule 在打开标签前冻结。
- Metrics：mixed-label exact classes、先无标签构建再检查方向的 cross-task mutual-nearest pairs、similarity distribution、empirical exact-signature conditional Bayes error、task coverage。
- Success criterion：至少 5 个 exact opposing pairs、覆盖 3 tasks；否则要求至少 20 个 cross-task mutual-nearest opposing pairs、公开相似度 ≥ 0.90，并明确只支持 approximate-representation claim。
- Failure：不得写“不可区分”；仅保留 tested classifiers are weak。
- Target：Figure 2；MUST-RUN。

### B4：Strong deployment-visible baseline

- Claim：actionability gap 不是只由弱线性分类器造成。
- Systems：已有 cross-task semantic classifier；冻结 prompt 的强 LLM judge；order-swapped A/B 复评；不访问 receipts/outcomes/references。
- Metrics：task-disjoint ROC-AUC、PR-AUC lift、balanced accuracy、Rescue/Reverse recall、swap consistency、calibration。
- Success interpretation：若强基线仍低且 collision Gate 通过，支持“tested public interface 下的 actionability gap”；若显著高，则论文改写为表示/训练目标问题。
- Target：Table 2；MUST-RUN。

### B5：Failure-family synthesis

- Claim：三类直观修复均未越过预注册非劣 Gate，且 credit correction 失败并非普遍梯度反向。
- Inputs：冻结的 static/one-step/two-step、DeltaGuard/Goal Query、RAPG/EditCredit/Counterfactual Credit Replay artifacts。
- Metrics：每类只保留最强版本；AUC、Reverse recall、Rescue drop、task bootstrap、oracle-gradient cosine、cost。
- Target：Table 3 和 Appendix；MUST-RUN，但仅重算旧 artifacts，不再调参。

### B6：Reusable Exact Shadow benchmark package

- Package：`public_events.jsonl`、加密/分离的 `private_outcomes.jsonl`、task-disjoint splits、schema、official evaluation CLI、integrity manifest/SHA-256、dataset card、license/provenance、minimal example。
- Public/private boundary：公开输入不含 evaluator state、reference action、hidden DB/context 或 branch outcomes；release 前核对 AppWorld/ToolSandbox license。
- Acceptance：从干净目录运行 verifier 能重建所有 paper tables；hash tampering 必须 fail closed。
- Target：Artifact contribution；MUST-RUN。

## Run Order

| Milestone | Runs | Decision gate | Estimated cost |
|---|---|---|---|
| M0 | 历史 hash inventory + freeze B1 | 零交集、完整 seal | CPU 5--15 min |
| M1 | B3 collision audit on frozen banks | exact/approx collision Gate | CPU 10--30 min |
| M2 | B5 synthesis + B6 package skeleton | tables reproducible、public/private clean | CPU 20--60 min |
| M3 | B2 fresh 13-scenario Exact Shadow | 双方向和 task coverage Gate | 约 1--3 h，取决于 API/ToolSandbox |
| M4 | B4 strong judge | task-disjoint + swap consistency | 约 15--45 min/API cost |
| M5 | paper table/figure freeze | 所有 paper 数字来自 sealed artifacts | CPU 10--30 min |

## Stop Rules

- B1 有任何旧 hash 重叠：停止，不把该批称为 fresh。
- B2 未同时出现 Rescue/Reverse：主张降级为已有开发集现象，不追加任务挽救。
- B3 不过：删除“indistinguishable”措辞。
- B4 明显成功：将贡献改写为弱表示/目标错配，而不是信息不足。
- 不再训练或命名新的 V10/V11 loss/router。

## Paper Targets

- Table 1：两 benchmark + fresh split 的 Rescue/Reverse/Zero、tasks、oracle headroom。
- Figure 1：聚合 Harness success 如何混合相反 effects。
- Figure 2：collision pairs 与 representation-conditional error lower bound。
- Table 2：static/strong-LLM/public baselines 的 task-disjoint actionability。
- Table 3：routing/probing/credit 三类冻结 Gate 的一行式负边界。
