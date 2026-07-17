# 可复现性检查

- [x] 官方数据 source commit 固定。
- [x] split seed、hash、原子 tool-OOD 零重叠和实际数量写入 manifest。
- [x] 训练 seed 可配置。
- [x] ground truth 来自冻结 reference actions / programmatic checker。
- [x] audit probability 在 draw 前持久化并可校验。
- [x] EMA 在当前估计之后更新。
- [x] shadow 与 failed replay 计入预算。
- [x] smoke 输出标为非论文证据。
- [ ] H200 单 seed pilot 已完成。
- [ ] 三 seed confirmatory 已完成。
