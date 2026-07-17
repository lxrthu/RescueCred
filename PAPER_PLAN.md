# RescueCredit 论文规划（草案）

**Working title**: *RescueCredit: Learning from Verified Runtime Corrections without Masking Policy Errors*  
**Alternative title**: *When Corrections Hide Errors: Counterfactual Credit Assignment for Tool Agents*  
**Target venue**: AAAI  
**Type**: 方法 + 受控实验  
**Date**: 2026-07-16  
**Page budget**: 7 technical-content pages（references 另计，提交前按当年 CFP 复核）  
**Section count**: 7

## 一句话贡献

在运行时纠错来源已经验证且对所有方法完全相同的条件下，RescueCredit 用稀疏 counterfactual Shadow 和 token provenance 恢复被纠错执行所掩盖的原始策略信用。

## 明确范围

- **本文研究**：给定 verified correction 后，如何进行 credit assignment。
- **本文不研究**：如何自动生成可靠 correction。
- **不得声称**：end-to-end reference-free Harness、自动纠错已解决、AppWorld 自动 Harness 可部署。
- **Harness 失败结果**：放入 Limitations/Appendix，说明原型 holdout precision 为 67.6%，未达到 90% gate，因此未用于主训练。

## Claims–Evidence Matrix

| Claim | 当前证据 | 状态 | 论文位置 |
|---|---|---|---|
| C1. Many-to-one runtime correction 会让好坏 policy proposal 获得相同 assisted return，形成 compensation trap。 | Exact Rescue-MDP、轨迹构造、需补充 Naive/Mask 的 credit-fidelity 对比。 | 部分支持，需主结果 | §3, §5.1 |
| C2. Commit-before-draw randomized residual estimator 在固定审计概率和 control variate 下无偏。 | 只对未裁剪 estimator 写无偏性；正式证明和假设待完成。 | 理论待完成 | §4.2 |
| C2b. Weight clipping 控制方差但引入偏差。 | 需要 bias/variance 分析；不得称 clipped estimator 无偏。 | 待分析 | §4.2/Appendix |
| C3. 在相同 frozen verified corrections 下，RescueCredit-v2 比 Mask+Correction 提升 S_off 或 First-pass。 | 尚无有效 AppWorld 主实验；API-Bank smoke 不能支持。 | **核心待验证主张** | §5.2 |
| C4. Sparse Shadow 能以低于 Full Shadow 的额外交互成本接近其收益。 | 已有成本日志机制；缺 matched multi-seed 实验。 | 待验证 | §5.3 |
| C5. 自动 reference-free Harness 足够可靠。 | 最终 27-task holdout precision 67.6% < 90%。 | **不支持；从主张删除** | §6 Limitation / Appendix |

## 0. Abstract（约 0.25 页，最后写）

- **Problem**：runtime correction 能提高 assisted task success，却把错误 proposal 隐藏在成功执行后面，使 policy 收到错误信用。
- **Approach**：给定 frozen verified corrections，RescueCredit 对首次 teachable intervention 建立 counterfactual shadow，估计原动作回报，并按 prefix/suffix/harness provenance 路由梯度。
- **Evidence placeholder**：必须填入 Mask+Correction vs RescueCredit-v2 的三 seed S_off、First-pass 和额外交互成本；没有主结果前不得写定量结论。
- **Scope sentence**：Correction generation is treated as an external verified component rather than a contribution of this work.
- **禁止**：使用 Harness development precision 或工程 smoke 作为摘要主结果。

## 1. Introduction（约 0.95 页）

### 叙事顺序

1. **Hook**：安全 tool agent 常在执行前后被 runtime guard/corrector 修改动作；成功执行并不意味着 policy proposal 正确。
2. **Failure mode**：bad proposal A 和 corrected proposal B 都得到 assisted success，标准 on-policy return 无法区分二者。
3. **Why existing masking is insufficient**：Mask+Correction 可以阻止错误 token 接收正向梯度，但不知道原动作如果真的执行会造成什么后果，也无法利用纠错是否真正救活轨迹。
4. **Key insight**：只在首次可教学纠错处做稀疏 counterfactual audit，用原动作 shadow return 决定 correction preference 的方向和权重。
5. **Scope boundary**：本文假设 correction 已由外部系统验证；所有方法共享同一 correction bank。
6. **Results preview**：留空，待 seed 42 gate 和 3-seed confirmatory results 后填写。

### Contributions（最终限制为 3 条）

1. 形式化 compensation trap：仅观察 corrected execution 时，many-to-one action mapping 使 assisted return 对 proposal credit 不可识别。
2. 提出 RescueCredit-v2：validity-gated、length-normalized correction preference，加上 sparse shadow-derived causal weighting；Shadow 不再污染整条 GRPO return。
3. 在 exact MDP 与受控 tool-agent benchmark 上，与 Naive H+GRPO、Mask+Correction、Full Shadow 做 matched-budget、multi-seed 比较，并报告 unassisted behavior 与真实 Shadow 成本。

### Hero Figure（Fig. 1）

- 左侧：Policy 提出错误动作 A；runtime corrector 输出验证后的 B；环境执行 B 并成功。
- 中间上方（Naive）：A/B 被同一个 assisted return 反向传播，A 被错误奖励。
- 中间下方（Mask）：A 被 mask，但错误原因和 counterfactual 后果丢失。
- 右侧（RescueCredit）：snapshot → 原动作 A 的 sparse Shadow → Δ；prefix 接收 original-action credit，B>A 或 A>B preference 由 Δ 和 validity 决定，Harness token 无 policy gradient。
- 图中明确标注：`same frozen verified correction source for every method`。
- **Caption draft**：Runtime correction can make an incorrect proposal and its corrected execution observationally equivalent. RescueCredit audits the original action on a sparse shadow branch and routes the recovered counterfactual signal only to policy-owned tokens, while all baselines receive the same verified corrections.

## 2. Related Work（约 0.55 页）

按问题组织，不逐篇罗列：

1. Tool-agent RL 与环境反馈。
2. Runtime guardrails、action correction、shielding 与 safe RL。
3. Credit assignment、counterfactual baselines、off-policy correction。
4. Preference optimization from corrected actions。

定位句：已有工作多研究如何获得或执行 correction；本文研究 correction 已存在时，如何避免其成功执行掩盖 proposal 的学习信号。

所有引用必须用 `/research-lit` 或已有 BibTeX 验证，不能凭记忆生成。

## 3. Problem Setup: The Compensation Trap（约 0.75 页）

- 定义 policy proposal $A_t$、verified correction $B_t$、execution mapping $H(A_t)$。
- 定义 assisted return $G_H$、original-action counterfactual return $G_0$。
- 定义 first teachable intervention、policy prefix/suffix、Harness/tool/environment provenance。
- 用两状态 Rescue-MDP 展示 $A_t \neq B_t$ 但执行轨迹和 $G_H$ 相同，导致 proposal credit 不可识别。
- Proposition 1：在 many-to-one correction mapping 下，仅观察 corrected execution return 无法识别原 proposal 的相对价值。
- 主文给 proof intuition；完整证明放 Appendix。

## 4. RescueCredit-v2（约 1.35 页）

### 4.1 Verified correction interface

- Correction bank 在训练前冻结。
- 必须披露 correction 的来源、冻结时间、hash、事件数、验证规则，以及是否使用 benchmark reference。
- 每条 record 包含可执行、语义有效、轨迹成功/失败等独立验证状态。
- Reference 或外部 verifier 只产生 correction record；Mask 与 V2 使用完全相同的 record。
- 主张限定为 conditional credit assignment，不声称 correction source reference-free。

### 4.2 Sparse causal audit

- Commit-before-draw 审计流程。
- 真实事件审计概率 $p_i$；warm-start 事件单独记录。
- Shadow 从 intervention 前 snapshot 执行原动作 A，并使用确定性 continuation 或记录随机性。
- $\Delta = G_{\text{assisted}} - G_{\text{shadow}}$。
- 证明/命题：固定 $p_i$ 与 control variate 时 residual estimator 无偏；报告方差与 clipping 影响。
- 无偏性只适用于 unclipped estimator；clipping 明确作为有偏的方差控制。
- Shadow steps 作为额外成本，不能与 main interaction budget 混合。

### 4.3 Validity-gated causal preference

- A/B 三值 semantic validity：true / false / unknown。
- $\Delta>0$：学习 B>A；$\Delta<0$：学习 A>B 并关闭普通 correction；$\Delta=0$：无 causal loss。
- 动作概率使用 mean action-token log-prob，避免长度偏差。
- 权重 $w=\min(|\Delta|/p_i,w_{\max})$。
- Shadow 只筛选和加权 correction preference，不修改整轨迹 GRPO reward。

### 4.4 Provenance routing

- Prefix token：original-action/counterfactual credit。
- Suffix token：assisted return。
- Harness、tool、environment token：zero policy gradient。
- Algorithm 1 放完整伪代码；日志字段支持复现和审计。

## 5. Experiments（约 2.15 页）

### 5.1 Experimental protocol

- **Environments**：Exact Rescue-MDP；API-Bank controlled 或 AppWorld frozen-correction subset（二者只能在 correction records 冻结并可审计后进入主表）。
- **Model**：Qwen2.5-7B-Instruct，LoRA，匹配 dtype、LR、interaction budget。
- **Baselines**：Naive H+GRPO、Mask+Correction、Full Shadow、RescueCredit-v2。
- **Seeds**：42 pilot；方法与参数冻结后 45/46/47 confirmatory。
- **Fairness**：相同 main environment steps；Shadow steps 额外报告；相同 correction bank；相同 evaluation tasks。
- 同时保证相同 correction events、训练 token 数，并尽可能增加 compute-matched 对照。
- **Primary metrics**：S_off、First-pass。
- **Secondary metrics**：S_on、dependence gap、intervention rate、correction fidelity、audit precision、extra-step ratio。
- 报告 mean ± std、单 seed 值、task-level paired bootstrap 或适当配对检验。

### 5.2 Main result: Does causal credit improve the policy?

- **Table 1**：四个方法 × S_off / First-pass / S_on / main steps / shadow steps。
- 核心 gate：RescueCredit-v2 必须在 S_off 或 First-pass 上优于 Mask+Correction；只提升 S_on 不支持论文主张。
- 如果只在一个 seed 上改善，不写“outperforms”；必须等待 confirmatory seeds。

### 5.3 Efficiency against Full Shadow

- **Fig. 2**：x 轴 extra shadow-step ratio，y 轴 S_off；点为不同 audit probability。
- 比较 $p\in\{0.1,0.2,0.4,1.0\}$，但只在 seed 42 选定最终 p；确认实验不再调。
- 目标主张：在显著低于 p=1 的成本下接近 Full Shadow；如果不成立则删除 C4。

### 5.4 Focused mechanism result

- 主文最多保留一个消融：$\lambda_{causal}=0$（即移除核心因果信号）；其余移 Appendix。
- 报告 nonzero causal events、B>A/A>B/zero-Δ 数量，避免只报最终成功率。

## 6. Analysis, Limitations, and Failed Harness Attempt（约 0.55 页）

- 自动 reference-free AppWorld Harness 最终 holdout：23/34 correct，precision 67.6%，未达到 90% gate；因此没有用于主训练。
- 这项失败结果只说明 automatic correction generation 未解决，不说明 conditional credit assignment 无效。
- 外部 verified corrections 的假设限制部署范围和可扩展性。
- Shadow 增加环境执行成本，并依赖精确 snapshot/rollback。
- 程序化 verifier 或 benchmark reference 可能限制生态有效性。
- 不把 0% observed harm 写成一般安全保证。

## 7. Conclusion（约 0.25 页）

- 重申 compensation trap 和 counterfactual credit routing，不复述摘要。
- 只总结被三 seed 主实验支持的结论。
- Future work：独立研究 multi-turn deployable Harness；本文不把它包装成已完成贡献。

## Figure and Table Plan

| ID | 类型 | 内容 | 数据源 | 优先级 |
|---|---|---|---|---|
| Fig. 1 | Hero/architecture | Naive vs Mask vs RescueCredit 的 credit flow | 方法示意，人工绘制 | HIGH |
| Table 1 | Main results | 四方法 S_off、First-pass、S_on、成本，3 seeds | 待运行 frozen-correction main experiments | HIGH |
| Fig. 2 | Pareto curve | Shadow cost vs S_off，不同 audit p | 待运行 p sweep | HIGH |
| Table A1 | Appendix ablations | validity、reversal、normalization、provenance | 待主结果通过后规划 | LOW |
| Fig. A1 | Appendix mechanism | preference margin、nonzero causal events、checkpoint S_off | training logs | LOW |
| Appendix Table | Negative result | automatic Harness 各版本及 final holdout | Harness audit JSON | LOW |

## Citation Plan

- §1：tool-agent learning、runtime correction/safety shielding、credit assignment动机；全部待检索验证。
- §2：按四类工作系统检索，不从记忆写作者、年份或 BibTeX。
- §3：many-to-one observation/identifiability、counterfactual credit 的理论背景；待验证。
- §4：importance sampling/control variates、preference optimization、provenance masking；待验证。
- §5：API-Bank、AppWorld、GRPO/Qwen 官方来源；优先论文与官方文档。

## Evidence Freeze and Go/No-Go

1. 先构建并冻结 verified correction bank；记录 hash、来源、validity、执行回执。
2. seed 42 跑 Mask+Correction 与 RescueCredit-v2；参数完全一致。
3. 若 V2 未改善 S_off 或 First-pass：停止，不能写方法优越性论文。
4. 若 gate 通过：冻结超参，跑 seeds 45/46/47。
5. 主结果稳定后才进入 ablation；`test_normal` / `test_challenge` 最终只开一次。

## 当前写作状态

- 可以先写：Introduction 问题定义、Compensation Trap、Method、实验协议、Limitations。
- 暂时不能写：Abstract quantitative result、主结果结论、优越性措辞、最终标题中的强结果暗示。
- 当前项目仍处于实验阶段，不是 submission-ready paper。

## 独立大纲复核

| 维度 | 分数 | 已采用的最小修复 |
|---|---:|---|
| 逻辑流 | 8/10 | 核心压缩为 verified B + sparse counterfactual A → B/A preference 的方向与权重。 |
| Claim–evidence 对齐 | 8/10 | 分离未裁剪无偏性与 clipping bias；禁止主结果前写 outperform。 |
| 实验完整性 | 3/10 | 明确 correction bank、seed 42、45/46/47、Full Shadow 与 paired uncertainty 为投稿最低要求。 |
| Positioning | 7/10 | 自动 Harness 降为限制/附录；主文突出 Mask 无法恢复的 counterfactual 信息。 |
| 7 页可行性 | 4/10 | 主文只留 Fig.1、Table 1、一个 cost–performance 图、一个 compact algorithm 和一个 focused ablation。 |

**Reviewer verdict**：当前 AAAI No-Go（高置信度）。只有 frozen correction bank 后，V2 在公平 seed-42 pilot 优于 Mask，并在 seeds 45/46/47 保持方向一致且 paired uncertainty 不与“无改进”相容，才转为 Go。
