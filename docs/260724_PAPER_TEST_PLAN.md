# Paper Test 总计划与完成清单

> 更新日期：2026-07-24  
> 用途：统一记录论文投稿前需要完成的实验、冻结配置、运行命令、产物和验收条件。  
> 详细操作与指标定义见 [260724_PAPER_DELAY_AWARE_EXPERIMENTS.md](260724_PAPER_DELAY_AWARE_EXPERIMENTS.md)。

## 1. 论文要验证的核心命题

论文实验应围绕以下问题组织，而不是继续无边界增加测试：

1. 忽略计算延迟的 learned MPC 是否会在真实执行中退化？
2. future alignment、execution re-anchor 和 fast feedback 能否恢复接近
   zero-delay Ideal 的性能？
3. Virtual fixed-delay 结论在真实 Threaded asynchronous planner 中是否仍成立？
4. 完整方法是否稳定优于 Naive delayed MPC 和 Direct IK？
5. 性能提升是否在多轨迹、多 CEM seed 和扰动下成立？
6. 计算延迟、planner rate、late packet、fallback、安全投影和控制 deadline 是否满足
   soft-real-time 仿真要求？
7. 剩余误差有多少来自 learned dynamics，而不是 delay-aware 控制结构？

## 2. Gate 0：先冻结唯一论文配置

任何正式实验开始前，必须先完成这一项。论文主表禁止混用 checkpoint、reference、
projection 语义或控制代码版本。

### 2.1 当前存在的配置分叉

现有 `outputs/paper_delay_aware` 使用：

- checkpoint：`outputs/checkpoints/gru_20260720_202923`；
- H20、128 samples、2 CEM iterations；
- virtual 每 5 tick 重规划；
- 标定 D7；
- 文档记录的正式语义为 `planner_projection=off`。

当前仓库默认配置则是：

- `planner_projection=on`；
- `planner_projection_backend=compiled`；
- `planner_projection_strategy=two_stage`，即方案 B。

现有 `outputs/paper_delay_aware/manifests/paper.json` 没有显式保存
`planner_projection`、backend、strategy 和部分 residual semantics 字段。因此旧
paper suite 不能作为“当前方案 B”的正式证据，也不能与新的方案 B rollout 混合。

### 2.2 已锁定的论文方法

论文最终方法固定为 compiled two-stage projection：新输出根目录为
`outputs/paper_delay_aware_two_stage_v1`，必须重新标定 D、构建 schema-v4 manifest，
并重跑全部 P0 控制实验。旧 `outputs/paper_delay_aware` 的 projection-off suite 仅保留
为先导/历史证据，禁止与新结果合并或用于主表。

### 2.3 冻结表

正式运行前在新 manifest 中逐项确认：

| 配置项 | 论文冻结值 |
|---|---|
| Git commit | 待冻结，必须 clean worktree |
| checkpoint / normalizer | `gru_20260720_202923`，或明确选择其他唯一版本 |
| reference manifest | 四条 immutable paper references 及 SHA-256 |
| history / horizon | 16 / 20 |
| samples / batch / iterations | 128 / 128 / 2 |
| virtual replan interval | 5 ticks，20 Hz |
| planner projection | `on` |
| projection backend / strategy | `compiled` / `two_stage` |
| residual / packet semantics | `requested` / `requested`；feasibility=`finite` |
| nominal semantics | `raw_ik` |
| feedback gains | `Kq=0.3`、`Kdq=0.015`，若不修改 |
| planner guard | 5 ms |
| control / model dt | 10 ms / 10 ms |
| GPU、driver、CUDA、PyTorch、MuJoCo | 必须记录 |

## 3. 实验优先级

| 级别 | 含义 |
|---|---|
| P0 | 主论文不可缺少；缺失时不建议投稿 |
| P1 | 强烈建议；用于回答常见审稿问题 |
| P2 | 高价值补充或诊断，不应延误主稿 |
| P3 | 开发历史，仅用于参数解释，不进入主结论 |

## 4. 预实验与可复现性检查

### T0：代码回归与环境记录（P0）

目的：证明正式结果来自可复现、通过测试的冻结代码。

```bash
export PAPER_OUT=outputs/paper_delay_aware_two_stage_v1

python -m unittest \
  mpc.test_paper_experiments \
  mpc.test_residual_mpc \
  mpc.test_asap_planner_worker \
  mpc.test_asap_timing \
  mpc.test_logging_threaded \
  mpc.model_c.test_oracle \
  mpc.test_robustness \
  mpc.test_model_a_robustness_evaluate \
  -v 2>&1 | tee "$PAPER_OUT/logs/unit_tests.log"
```

验收：

- 测试全部通过；
- `git status --short` 为空；
- 保存 commit、checkpoint/reference hash 和软件/硬件版本；
- 不在跑正式 threaded 实验时运行其他 GPU 任务。

### T1：独立 GRU 验证（P0）

目的：报告 one-step 和 1/5/10/20-step open-loop prediction，证明 frozen dynamics
具有有限 horizon 内的预测能力。

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" validate-model \
  --num-rollouts 20 --rollout-len 200
```

必须报告：

- q/dq RMSE、NMSE、R²、amplitude ratio；
- 1/5/10/20-step error；
- divergence rate；
- evaluation rollout 未参与训练的说明。

当前旧 paper root 已有 20 个 rollout；若 Gate 0 选择新配置和相同 checkpoint，可核验
hash 后复用模型验证，控制实验仍必须使用新的 manifest。

### T2：生成独立标定与正式 references（P0）

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" generate-calibration-reference

python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" generate-references
```

正式轨迹：

- circle；
- figure-8；
- fast ellipse；
- rounded square。

验收：

- reference 有足够的 `H + max(D) + preview` padding；
- 保存 execution steps、速度 profile、IK 误差和 SHA-256；
- calibration reference 不进入正式测试。

### T3：真实 Threaded E2E 延迟标定（P0）

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" calibrate-delay \
  --samples 500 --provisional-delay 10 --guard-ms 5
```

冻结：

```text
D_cal = ceil((P95(snapshot-to-publication E2E) + 5 ms) / 10 ms)
```

验收：

- 至少 500 个有限 E2E 样本；
- 记录 P50/P95/P99/max、late count 和标定平台；
- checkpoint、H、CEM budget、projection 策略变化后必须重新标定；
- Ideal 使用 D0，其余正式 delay-aware 方法共同使用同一个 `D_cal`。

### T4：Preview IK 独立标定（P1）

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" calibrate-preview \
  --preview-values 0,1,2,3,4
```

选择规则：只在独立 calibration reference 上选最小 TCP RMSE；并列时选较小 preview。
禁止在正式测试轨迹上分别调 preview。

### T5：Smoke test（P0）

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" smoke
```

覆盖 Direct IK、Ideal、Naive、FullVirtual、Threaded、NoFutureAlignment、
NoReanchor 和 NoFeedback。Smoke 只验证数据流，不能写入论文结果。

## 5. 主论文实验

### E1：五控制器主比较（P0）

目的：回答完整方法是否接近 Ideal，并优于 Naive 和 Direct IK。

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" run --suite main --resume
```

矩阵：

| 方法 | 轨迹 | seed | Case 数 |
|---|---:|---:|---:|
| Direct IK | 4 | deterministic | 4 |
| IdealZeroDelay，D0 | 4 | 5 | 20 |
| NaiveDelayed，D_cal | 4 | 5 | 20 |
| FullVirtual，D_cal | 4 | 5 | 20 |
| ThreadedASAP，D_cal | 4 | 5 | 20 |
| **总计** |  |  | **84** |

主指标：

- TCP position RMSE/P95/max；
- joint RMSE、orientation RMSE；
- paired `method − Ideal`、`method − Direct IK`；
- planner failure、fallback、late packet、packet expiration；
- command velocity/acceleration 和 joint-limit violation；
- Threaded solve latency、E2E latency、planner rate、control period/jitter/deadline。

验收：

- 84/84 唯一 case；
- MPC 的 5 seeds 是相同 trajectory 下的配对 CEM seeds；
- bootstrap 单位是 trajectory-seed，不是 10 ms tick；
- Virtual wall time 不解释为实时部署性能；
- 安全违例和 worker fatal 为 0。

### E2：核心机制消融 / causal ablation（P0）

目的：分别检验 future alignment、execution re-anchor 和 fast feedback。

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" run --suite ablation --resume
```

| 方法 | 被移除机制 | Case 数 |
|---|---|---:|
| FullVirtual | 无，复用 E1 | 20 |
| NoFutureAlignment | future-state/reference alignment | 20 |
| NoReanchor | execution-time reconciliation | 20 |
| NoFeedback | fast state feedback | 20 |

表格共 80 行，但 FullVirtual 复用主实验，只新增 60 个 rollout。

必须报告：

- `NoX − FullVirtual` 的逐轨迹均值和 paired bootstrap CI；
- 不把三个差值解释为可相加的独立贡献；
- absolute-command 消融的实际速度/加速度违例；
- safety projection 修改量和 planner/execution command discrepancy。

**当前状态：缺失。** 现有 paper root 只有三项机制的 smoke，没有
`ablation.csv`、`ablation_aggregate.csv` 或
`ablation_paired_bootstrap.json`。这是投稿前最优先补齐的实验。

### E3：Direct IK 与 Preview IK（P1）

目的：排除 MPC 只是在补偿纯时间相位滞后的解释。

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" run --suite preview --resume
```

Preview IK 单独成表，不加入五控制器主表。必须同时报告 raw Direct IK、统一 preview
值及其独立标定过程。

## 6. 延迟与模型误差实验

### E4：Delay sweep（P1）

目的：展示 Full 与 Naive 对延迟增加的敏感性，并计算 latency recovery。

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" run --suite delay_sweep --resume
```

```text
protocols: Full, Naive
trajectories: circle, fast_ellipse
D: 0, 2, 4, 6, 8
seeds: 0, 1, 2
总计: 60 cases
```

必须绘制 TCP RMSE vs D，并报告：

```text
latency recovery = (E_naive - E_full) / (E_naive - E_ideal)
```

当前旧 paper root 已完成 60/60，但仅可用于其冻结的 projection-off 配置。

### E5：Oracle dynamics upper bound（P2）

目的：区分 learned dynamics 误差和 delay-aware 控制结构误差。

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" run --suite oracle --resume
```

```text
LearnedFull / MuJoCo Oracle
circle / fast ellipse
3 paired seeds
总计 12 cases
```

必须称为 dynamics upper bound，不能称为保持动作不变的纯模型误差消融。

## 7. 鲁棒性与安全实验

### E6：无扰动完整轨迹复核（P0）

目的：确认主结论在完整多圈轨迹上成立，而不是只由短前缀决定。

最低要求：

- circle、figure-8、ellipse、square；
- 至少 3 paired seeds；
- Direct、Ideal、Naive、Virtual、Threaded；
- 与 E1 相同 checkpoint、projection 语义和 D 标定。

现有旧 GRU 结果记录在
[260724_FOUR_MPC_ARCHITECTURES_H20_LEGACY_GRU.md](260724_FOUR_MPC_ARCHITECTURES_H20_LEGACY_GRU.md)，
可作开发复核，但 checkpoint 为 `gru_20260717_152930`，不能替代 final paper
checkpoint 的正式结果。

### E7：模型失配和外扰鲁棒性（P1）

建议至少包含：

- payload；
- actuator gain mismatch；
- observation noise；
- force pulse；
- nominal、medium、high 三档；
- Direct IK、Ideal、Naive、Virtual、Threaded；
- 相同 case 的 paired comparison。

必须报告 failure、恢复时间、force peak/integrated error、fallback、late packet、
安全违例和 tracking degradation。现有 H20 方案 B 鲁棒性结果见
[260724_DELAY_AWARE_MPC_ROBUSTNESS.md](260724_DELAY_AWARE_MPC_ROBUSTNESS.md)，但使用 checkpoint
`gru_20260717_182930`。若将其放入主论文，必须明确为独立 robustness study；若要
与 E1 数值直接合并，则必须用 final paper checkpoint 和完全相同配置重跑。

### E8：安全与实时性验收（P0）

所有正式 Threaded case 汇总检查：

- NaN/Inf、worker fatal、planner failure；
- joint/velocity/acceleration violation；
- control deadline miss；
- late drop、packet expiration、fallback；
- solve 和 E2E P50/P95/P99/max；
- planner Hz、control-period P99、start jitter；
- safety projection activation/offset；
- worst seed，而不只报告平均值。

论文只能声称 Python/MuJoCo 进程内的 soft-real-time 结果，不能声称真机硬实时。

## 8. 参数与实现选择证据

### E9：H、samples 和 iterations（P2）

已有旧 Ideal sweep 覆盖 H20/H25/H30、32/64/128 samples 和 2/3 iterations，可用于
说明历史参数选择；数据位于 `outputs/mpc/cem_horizon_grid_gru/`。

限制：

- 使用旧 checkpoint `gru_20260717_152930`；
- 属于旧同步 zero-delay MPC；
- 不是当前方案 B 或 Threaded 条件下的全参数搜索。

论文可据此说明 `H20 + 128×2` 是精度/延迟折中，但不得声称它是当前架构的全局最优。
若审稿目标强调参数最优性，应在 final paper checkpoint 下补一个小型 matched
sensitivity test。

### E10：Planner projection 选择（P0）

由于论文最终使用 two-stage projection，必须在 final checkpoint、references 和新 delay
calibration 下至少比较：

- projection off；
- full exact projection；
- compiled two-stage projection。

固定 H20、相同 D 标定规则、至少 circle + figure-8、3 paired seeds，同时报告
tracking、solve/E2E latency、planner rate、late packet 和安全投影。历史不同
checkpoint 的 projection 实验只能作为开发证据。

## 9. 汇总、统计和图表

### 9.1 统一重建 summary

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  summarize --suite all --bootstrap-samples 10000
```

必须存在：

```text
summaries/main.csv
summaries/main_aggregate.csv
summaries/main_paired_bootstrap.json
summaries/latency_recovery.json
summaries/ablation.csv
summaries/ablation_aggregate.csv
summaries/ablation_paired_bootstrap.json
summaries/delay_sweep.csv
summaries/delay_sweep_aggregate.csv
summaries/preview.csv
summaries/preview_aggregate.csv
summaries/oracle.csv
summaries/oracle_aggregate.csv
```

### 9.2 最低论文图表

主文建议保留：

1. 控制架构图：100 Hz executor、background planner、future anchor 和 packet activation；
2. 五控制器主结果表；
3. 代表性 circle / fast-ellipse TCP tracking 图；
4. causal ablation 配对差值图；
5. TCP RMSE vs delay D；
6. Threaded latency distribution、planner rate 和 late/fallback 图；
7. GRU multi-step prediction 图或表。

补充材料：

- 四轨迹逐 seed 完整表；
- command smoothness、torque 和 safety projection；
- Preview IK；
- Oracle upper bound；
- robustness conditions；
- H/sample/iteration 和 projection 参数敏感性。

### 9.3 统计规则

- 统计单位是 trajectory-seed case；
- 同一 reference 和 seed 做 paired difference；
- 主表报告 mean、standard deviation 和 95% bootstrap CI；
- 同时报告 worst trajectory / worst seed；
- Direct IK 是确定性的，不把重复 seed 当成额外环境随机性；
- 不以每个 10 ms tick 作为独立样本；
- 不只报告 pooled mean，必须给出分轨迹结果；
- 完整三圈与 500-step prefix 必须分表，禁止混算。

## 10. 当前完成状态

以下仅描述 `outputs/paper_delay_aware` 旧 projection-off paper root：

| 项目 | 期望 | 当前 | 状态 |
|---|---:|---:|---|
| Delay calibration | 500 samples | 已生成，D7 | 完成 |
| Preview calibration | 5 candidates | 已生成，选中 preview 4 | 完成 |
| GRU validation | 20 rollouts | 20 | 完成 |
| Smoke | 8 | 8 | 完成 |
| Main | 84 | 84 | 完成 |
| Causal ablation | 80 table / 60 new | 0 formal | **缺失** |
| Delay sweep | 60 | 60 | 完成 |
| Preview | 4 | 4 | 完成 |
| Oracle | 12 | 12 | 完成 |

由于 Gate 0 的 projection 配置分叉，如果论文选择当前方案 B，上表中的控制实验应视为
旧版本先导结果，需要在新 output root 下重新执行，而不是覆盖。

## 11. 投稿前最终验收

- [ ] 唯一 final checkpoint 和 normalizer 已冻结；
- [ ] 唯一 clean Git commit 已记录；
- [ ] projection 和 residual semantics 已写入 manifest；
- [ ] references、checkpoint、normalizer 均有 SHA-256；
- [ ] 重新标定 D，且配置与正式 planner 完全一致；
- [ ] Main 84/84；
- [ ] Causal ablation 80/80 表格 case；
- [ ] Delay sweep 60/60；
- [ ] Preview IK 和 GRU validation 完整；
- [ ] 所有 P0 safety/real-time 检查通过；
- [ ] 主表同时给出分轨迹与 pooled 结果；
- [ ] paired bootstrap 和 worst-seed 已报告；
- [ ] 不混用旧 GRU、robustness GRU 和 paper GRU；
- [ ] 不混用完整三圈与 500-step prefix；
- [ ] Virtual 与 Threaded 的计时含义已分开解释；
- [ ] limitation 明确仅为 MuJoCo soft real-time，无真机硬实时结论；
- [ ] 所有图表可由冻结 CSV/JSON 重新生成；
- [ ] 文档、代码和 manifest 已提交，工作区干净。
