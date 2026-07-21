# EditCredit Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| EC000 | M0 | canonical AST edit span 单测 | Edit extractor | synthetic | exact changed fields | MUST | DONE_LOCAL | structural tests passed |
| EC001 | M0 | 错误信用梯度复现 | Naive B-credit | synthetic | A contamination grad | MUST | READY_SERVER | bound autograd Gate before freeze |
| EC002 | M0 | firewall 梯度验证 | Mask | synthetic | A contamination grad | MUST | READY_SERVER | bound autograd Gate before freeze |
| EC003 | M0 | edit-local 有符号信用验证 | EditCredit | synthetic | changed/unchanged grad, swap invariance | MUST | READY_SERVER | B1 hard precondition |
| EC010 | M1 | 强基线复算 | Default-B Mask | V4.4 126 pairs, task OOF | Rescue/Reverse/balanced/task-macro | MUST | READY_SERVER | no learned routing |
| EC011 | M1 | 当前方法复算 | Full-action signed preference | 同 EC010 | 同 EC010 | MUST | READY_SERVER | existing strongest direct baseline |
| EC012 | M1 | 主方法 seed-42 | EditCredit | 同 EC010 | primary + shortcut diagnostics | MUST | READY_SERVER | edit-local learner |
| EC013 | M1 | 主方法 seed-42 | EditCredit + Rescue constraint | 同 EC010 + inner calibration | primary + shortcut diagnostics | MUST | READY_SERVER | five-fold OOF Gate |
| EC020-022 | M2 | 稳健性 | EditCredit + constraint | seeds 42/43/44 | mean, std, task bootstrap CI | MUST_IF_GATE | BLOCKED | EC013 通过后解锁 |
| EC030 | M3 | fresh rollout | Mask | new preregistered tasks | unassisted/official/Rescue | MUST_IF_GATE | BLOCKED | 不得使用旧 confirmation 选择配置 |
| EC031 | M3 | fresh rollout | Full-action preference | 同 EC030 | 同 EC030 | MUST_IF_GATE | BLOCKED | matched cost |
| EC032 | M3 | fresh rollout | EditCredit + constraint | 同 EC030 | 同 EC030 | MUST_IF_GATE | BLOCKED | paper-facing only if passes |
| EC040-043 | M4 | 组件消融 | no firewall/full action/no randomization/no constraint | fresh development | primary deltas | NICE | BLOCKED | M3 正向后再跑 |
