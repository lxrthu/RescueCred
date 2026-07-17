# RescueEvent schema

`rescuecredit/types.py::RescueEvent` 是持久化契约。每个事件必须保存：运行/episode/group/candidate 标识、干预前 `state_ref/state_hash`、原提案和执行动作、patch/version、verifier 结论、permanent/teachable 安全边界、token spans、`p_i`、audit draw、`mu`、shadow return、`G0_hat` 与 rescue gain。

只有第一处改变可达路径的 teachable intervention 进入主估计。后续事件仍可记录，但不创建组合 shadow 分支。

