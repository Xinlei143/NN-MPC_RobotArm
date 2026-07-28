# UR5e 第二机械臂验证结果

本报告记录 UR5e 在独立模型、独立数据集、独立 GRU 和独立延迟标定下的正式 MPC
测试。它回答的是同一套延迟感知 residual CEM-MPC 控制架构能否在第二个 6-DoF MuJoCo
机械臂上复现；它不测试一个 GRU 在 ABB 与 UR5e 之间的迁移能力。

正式运行的控制代码冻结在 commit `abe107c`。随后提交的 `e7a7889` 只更新了说明文档。
正式 manifest、每个 run 的 fingerprint 和汇总文件都保存在
`outputs/paper_ur5e_v2/`。

## 实验设置

UR5e 使用 `configs/robots/ur5e.yaml`，控制周期为 10 ms，控制接口为绝对关节位置参考。
模型为 `gru_20260728_000313`，history length 为 16。与 ABB 正式配置相同的 MPC 设置为：
horizon 20、128 个 CEM samples、2 次 CEM iteration。两台机器人使用相同的四条
task-space 任务定义：circle、figure8、fast_ellipse 和 rounded_square。

同一任务不能直接复用 ABB 的关节 reference。UR5e 的 TCP、home pose、运动学分支和
joint limit 都不同，因此在相同任务尺度、速度和重复次数下，为 UR5e 单独求解 IK 并
生成 reference。跨平台比较的是 TCP 任务，而不是逐关节轨迹。

UR5e 通过 500 个有效 planner sample 做独立延迟标定：

| 指标 | 数值 |
| --- | ---: |
| snapshot-to-publication P50 | 51.61 ms |
| snapshot-to-publication P95 | 56.94 ms |
| guard | 5.00 ms |
| 采用的 anticipation delay | 7 steps（70 ms） |

延迟按 `ceil((P95 + guard) / control_dt)` 计算。不能把 ABB 的 D=6 直接复制给 UR5e。

## 覆盖范围与工程检查

正式矩阵包括四条轨迹、四个 MPC variant、每种 5 个 seed，以及每条轨迹一个确定性的
Projected Direct IK baseline，共 `4 + 4 × 4 × 5 = 84` 个 case。MPC variant 分别为
IdealZeroDelay、NaiveDelayed、FullVirtual 和 ThreadedAsync。

在正式运行前，Projected Direct IK 与四个 MPC variant 的短闭环 preflight 全部通过。
84 个 run 均有 `rollout.npz`、`run_fingerprint.json` 和 `run_summary.json`；实际状态数组
没有 NaN。汇总后的 safety 计数为：

| 检查项 | 结果 |
| --- | ---: |
| planner failure | 0 |
| joint-limit violation | 0 |
| command velocity violation | 0 |
| command acceleration violation | 0 |

ThreadedAsync 在长序列中记录到 15 次 late packet drop 与 8 次 control deadline miss。
它们是时序诊断事件，不是安全违例：本轮没有触发规划失败，也没有产生超出关节、速度或
加速度限制的执行命令。后续部署到真机前，仍应在目标硬件上重新标定 D，并把这些时序指标
作为持续监控项。

## UR5e 跟踪结果

下表是各 method 在其正式 case 上的平均结果。TCP 指标以毫米计；Projected Direct IK
只有每条轨迹一个确定性 run，因此不与每种 20-case 的 MPC 结果做 seed 配对推断。

| 方法 | case 数 | TCP RMSE | TCP P95 | joint RMSE |
| --- | ---: | ---: | ---: | ---: |
| Projected Direct IK | 4 | 15.06 mm | 29.76 mm | 0.0211 rad |
| IdealZeroDelay | 20 | 9.53 mm | 20.96 mm | 0.0118 rad |
| NaiveDelayed | 20 | 34.78 mm | 72.41 mm | 0.0486 rad |
| FullVirtual | 20 | 10.43 mm | 22.27 mm | 0.0129 rad |
| ThreadedAsync | 20 | 10.42 mm | 22.42 mm | 0.0133 rad |

IdealZeroDelay 是没有真实异步延迟的诊断上界，不能当作部署性能。真正需要比较的是
NaiveDelayed、FullVirtual 和 ThreadedAsync：延迟感知的 FullVirtual 将平均 TCP RMSE
从 34.78 mm 降到 10.43 mm。ThreadedAsync 的均值为 10.42 mm，与 FullVirtual 基本相同。

ThreadedAsync 的每 case 平均 timing 汇总为 E2E P50 49.12 ms、E2E P95 54.43 ms，planner
rate 22.58 Hz。这些值来自本次 GPU 与系统负载；它们不是 UR5e 控制器的固定属性。

## 平台内配对统计与 ABB 对照

`multi_robot_summary.py` 对每台机器人分别按 trajectory-seed 配对，并进行 10,000 次
bootstrap。它没有将 ABB 与 UR5e 的绝对误差合并为一个显著性检验。结果如下：

| 机器人 | 比较 | 匹配 case | 平均 TCP RMSE 差 | 95% CI |
| --- | --- | ---: | ---: | --- |
| ABB IRB2400 | FullVirtual − NaiveDelayed | 20 | -39.55 mm | [-42.02, -36.84] mm |
| UR5e | FullVirtual − NaiveDelayed | 20 | -24.36 mm | [-26.08, -22.64] mm |
| ABB IRB2400 | ThreadedAsync − FullVirtual | 20 | +0.05 mm | [-1.14, +1.27] mm |
| UR5e | ThreadedAsync − FullVirtual | 20 | -0.01 mm | [-0.39, +0.41] mm |

两个平台都显示出同一方向：忽略延迟的 NaiveDelayed 明显更差；在本次测试中，异步
ThreadedAsync 没有显示出相对 FullVirtual 的稳定精度损失。ABB 与 UR5e 的改善幅度不同，
这很正常，因为两者的机构、home pose、可达工作区和动力学模型都不同。这里能够支持的
结论是架构在两个独立标定的平台上均可运行，不是两台机器人的绝对跟踪误差相同，也不是
单一模型已经跨机器人泛化。

## 与本次实现问题的关系

UR5e 标定曾暴露出两个实现问题：旧的 preflight 读取了过期的 `states` 字段；旧的
`torch.compile(fullgraph=True)` 投影器会随 CEM population shape 反复编译，并在长标定
中达到 Dynamo 的 256 次重编译上限。正式结果使用修复后的状态接口和可复用的 TorchScript
投影图。指南已在 `docs/guides/ur5e-end-to-end-workflow.md` 中记录恢复方式。

## 可追溯产物

- UR5e manifest：`outputs/paper_ur5e_v2/manifests/paper.json`
- 延迟标定：`outputs/paper_ur5e_v2/calibration/delay.json`
- preflight：`outputs/paper_ur5e_v2/preflight/report.json`
- case index：`outputs/paper_ur5e_v2/runs/indexes/main.json`
- UR5e case-level summary：`outputs/paper_ur5e_v2/summaries/main.csv`
- UR5e aggregate 与 bootstrap：`outputs/paper_ur5e_v2/summaries/`
- ABB/UR5e effect sizes：`outputs/multi_robot_summary_ur5e_v2/effect_sizes.csv`
- ABB/UR5e 图：`outputs/multi_robot_summary_ur5e_v2/multi_robot_effect_sizes.png`
