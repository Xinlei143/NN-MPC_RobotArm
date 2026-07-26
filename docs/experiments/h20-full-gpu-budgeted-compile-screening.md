# H20 full residual：GPU budgeted task cost 的 compile 开关筛选

## 目的

本实验只比较 `torch.compile` 对 H20/full-residual、GPU budgeted stage-one task-space
cost 的端到端影响。它回答的是：

> 在相同的 Paper task reference、相同的 delay-aware 配置下，打开 compile 是否能降低
> planner 延迟，且不造成明显的闭环 tracking 损失？

本版本替代此前使用 `outputs/robustness/references/*_00` 短采集轨迹的 screening。
旧 reference 仅执行 500 steps，且速度显著更高，不能与 Paper robustness 的 nominal
结果直接比较。

> **默认配置状态**：本文件记录的是可选 GPU-FK 消融；当前默认方法是
> [Paper H20 residual MPC（stage-one GPU FK=off）](../Paper_test/mpc-four-architectures-robustness-results.md)。

## Reference 与实验协议

本轮直接使用 Paper robustness study 的 reference 文件：

| 轨迹 | Reference | 执行步数 | 执行时长 |
| --- | --- | ---: | ---: |
| Circle | `outputs/paper_delay_aware_two_stage_v1/references/circle/reference.npz` | 1807 | 18.07 s |
| Figure-8 | `outputs/paper_delay_aware_two_stage_v1/references/figure8/reference.npz` | 1807 | 18.07 s |

两个 reference 均使用论文中的 2 s approach、3 个 lap 和 return，因此与
[四种 MPC 架构 robustness 结果](../Paper_test/mpc-four-architectures-robustness-results.md)
的 nominal reference 定义一致。

本轮测试矩阵为：

```text
compile off/on × Circle/Figure-8 × seed 0/1 = 8 threaded rollouts
```

除 `stage_one_task_compile` 外，两个分支配置相同：

| 项目 | 配置 |
| --- | --- |
| Runner / delay protocol | `threaded_asap` / `full` |
| Delay | 共同 `D=6` ticks |
| Rollout horizon | H=20 |
| residual 参数化 | `full`，每个 10 ms tick 直接搜索 residual |
| CEM | 128 samples，2 iterations，batch=128 |
| Stage one | iteration 1 仅 base cost；iteration 2 加 GPU task cost |
| GPU task 点 | `[0, 3, 6, 9, 12, 15, 19]`，`nearest_interval` weighting |
| Final pool | exact projection + learned rollout + 完整 H20 MuJoCo FK task cost |
| Projection | on；backend=`compiled`；strategy=`two_stage` |
| Learned model | `gru_20260717_182930/best_model.pt` |

GPU budgeted 的语义为：最后一次 CEM iteration 的 128 条候选均使用 sparse-7 GPU
task cost 更新 elite 分布；最终执行候选仍由 MuJoCo exact final-pool scorer 决定。

## 延迟标定语义

本轮固定使用共同 `D=6`，以隔离 compile 的影响。该值来自同一 H20/full/GPU-budgeted
配置的既有 compile-on 标定：

- solve P95：39.83 ms；
- E2E P95：51.16 ms；
- `D = ceil((P95(E2E) + 5 ms) / 10 ms) = 6`。

标定文件：[h20_full_gpu_budgeted_delay.json](../../outputs/calibration/h20_full_gpu_budgeted_delay.json)。
本轮 Paper references 上的实测 E2E P95 也仍低于 55 ms 的 D=6 边界；但 compile-off
尚未用 Paper references 单独完成 500-plan calibration，故这仍是共同-D 筛选，而非
各自 calibrated-D 的部署对比。

## 分轨迹结果：误差与计算延迟

每个“轨迹 × compile 模式”包含 seed 0、1 两个 run。RMSE 是两个 run 的平均；TCP/姿态
P95 是先在每个 rollout 的 1807 个控制 tick 上计算 P95，再对两个 run 平均。solve/E2E
P95 同样先在每个 run 内计算，再在 seeds 间平均。

| 轨迹 | 模式 | TCP RMSE (mm) | TCP P95 (mm) | 姿态 RMSE (°) | 姿态 P95 (°) | Solve P95 (ms) | E2E P95 (ms) | Planner Hz | Late packet (%) | Failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Circle | compile off | 31.00 | 55.73 | 1.496 | 2.821 | 43.61 | 54.71 | 22.61 | 4.035 | 0 |
| Circle | compile on | 30.91 | 59.36 | 1.530 | 3.007 | 41.68 | 52.74 | 23.53 | 1.179 | 0 |
| Circle | on − off | −0.10 | +3.64 | +0.034 | +0.186 | −1.92 | −1.97 | +0.92 | −2.856 pp | 0 |
| Figure-8 | compile off | 22.97 | 49.63 | 1.065 | 2.112 | 42.80 | 53.84 | 23.11 | 2.273 | 0 |
| Figure-8 | compile on | 28.49 | 56.77 | 1.403 | 2.614 | 41.68 | 52.66 | 23.57 | 0.474 | 0 |
| Figure-8 | on − off | +5.52 | +7.14 | +0.338 | +0.502 | −1.12 | −1.18 | +0.46 | −1.799 pp | 0 |

四个 run 的跨轨迹总览仅用于阅读便利，不能替代上表：

| 模式 | TCP RMSE (mm) | TCP P95 (mm) | 姿态 RMSE (°) | 姿态 P95 (°) | Solve P95 (ms) | E2E P95 (ms) | Planner Hz | Late packet (%) | Planner failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compile off | 26.99 | 52.68 | 1.280 | 2.467 | 43.20 | 54.27 | 22.86 | 3.154 | 0 |
| compile on | 29.70 | 58.07 | 1.467 | 2.810 | 41.68 | 52.70 | 23.55 | 0.826 | 0 |
| on − off | +2.71 | +5.38 | +0.186 | +0.343 | −1.52 | −1.57 | +0.69 | −2.327 pp | 0 |

所有 8 条 rollout 均无 planner failure、joint-limit violation、command velocity violation
或 command acceleration violation。

## 与 Paper nominal 的量级核对

该表不是方法优劣比较，只用于核对 reference/运行时长已对齐。Paper robustness 中
ThreadedAsync nominal 使用 5 seeds，且不含本轮的 GPU budgeted stage-one refinement；
因此数值不应逐点一致。

| 轨迹 | Paper ThreadedAsync nominal TCP RMSE | 本轮 compile off | 本轮 compile on |
| --- | ---: | ---: | ---: |
| Circle | 28.40 mm | 31.00 mm | 30.91 mm |
| Figure-8 | 28.16 mm | 22.97 mm | 28.49 mm |

新结果已处于相同误差量级；此前短轨迹结果的 50–60 mm 不能再被用作本方案的性能估计。

## 解释

### 延迟

compile on 在两条轨迹上都降低了延迟：

- Circle 的 solve/E2E P95 分别降低 1.92/1.97 ms，late packet rate 降低 2.856 pp；
- Figure-8 的 solve/E2E P95 分别降低 1.12/1.18 ms，late packet rate 降低 1.799 pp。

这使 compile-on 的平均 E2E P95 为 52.70 ms，较 compile-off 的 54.27 ms 低 1.57 ms。
两者都保持 D=6，不过 compile-off 仅比 55 ms 边界低约 0.73 ms，余量较小。

### Tracking

在仅有两个 seed 时，compile on 没有显示稳定的 tracking 改善：

- Circle 的 RMSE 基本持平（−0.10 mm），但 TCP P95 增加 3.64 mm；
- Figure-8 的 RMSE/P95 分别增加 5.52/7.14 mm；
- pooled TCP RMSE/P95 分别增加 2.71/5.38 mm。

`torch.compile` 不应改变 FK 或 task-cost 的数学定义。这里的 tracking 差异更可能来自
threaded 系统中的 wall-clock 调度、packet 激活时刻与随机 CEM 采样/线程时序；尤其是
compile-on 的两个 seed 离散度较大。因此，当前结果支持“compile on 是延迟优化候选”，
但不支持其改善或稳定损害 tracking 的结论。

## 当前结论与下一步

1. **reference 已对齐**：本 screening 现在可与 Paper nominal 的 reference 定义比较；
2. **延迟收益**：compile on 在 Circle 与 Figure-8 上均降低 solve/E2E P95，并减少 late packet；
3. **tracking 尚未验收**：需要扩大 seeds 后才能判定 non-inferiority；
4. **下一步**：使用本文件的两个 Paper references 扩展至至少 5 个 seeds；再分别为
   compile off/on 做 500-plan calibration，并使用各自 calibrated D 复测。

## 结果文件

- [新 Paper-reference 输出](../../outputs/experiments/h20_full_gpu_budgeted_compile_paper_refs)
- [D=6 标定](../../outputs/calibration/h20_full_gpu_budgeted_delay.json)
- [Paper robustness 结果](../Paper_test/mpc-four-architectures-robustness-results.md)
