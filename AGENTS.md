# RescueCredit project instructions

## Pipeline Status

language: zh
stage: implementation

## Research integrity

- 评测必须对照数据集 ground truth 或程序化 checker，禁止用另一个模型的输出当标签。
- `outputs/smoke/` 只用于工程验证，不能作为论文证据。
- Shadow steps、failed replay steps 必须计入交互预算。
- 当前只归因第一处改变可达路径的 teachable intervention。
- `.env` 是唯一密钥入口；不得把密钥写入代码、配置或日志。

