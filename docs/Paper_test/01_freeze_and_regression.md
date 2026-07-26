# 代码、配置与回归测试

冻结提交：

- paper control commit：`8132559`
- analysis base commit：`4d6b6b5`
- 既有证据文档纳入版本控制：`77505a5`

schema-v6 显式冻结 full residual、stage-one task cost/compile off、lowest-cost execution、batch 128、preview nominal 0、MuJoCo FK exact pool、control dt 0.01 s 和 5-tick replan。

新增协议与 suite：

- `anchor_only`：future state/reference on，re-anchor off，feedback off；
- `delay_sweep_components`：96 cases；
- `projection_choice`：36 cases；
- projection 非劣界在运行前固定为 TCP RMSE 相对恶化不超过 5%。

最终回归命令对 `mpc/tests` 全量执行，结果为：

```text
130 passed, 1 skipped, 12 subtests passed
```

被跳过的测试由既有可选条件触发，不是失败。另完成 GPU workflow smoke。独立 replay 工具随后补了与主 runner 一致的 `dynamics_modeling` 模块路径；该修复只影响补充分析入口，不改变控制代码、manifest 或 rollout。
