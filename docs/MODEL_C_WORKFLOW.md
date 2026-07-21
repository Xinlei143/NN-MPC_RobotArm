# Model C：MPC-induced 数据闭环操作手册

本文给出从 Model A 开始，完成 Round 1、C1、Round 2、C2 和最终盲测的完整流程。请在仓库根目录执行全部命令，并在每一步确认输出完成后再进行下一步。

流程为：

```text
Model A
  -> 延迟标定
  -> Round 1（固定比例的 MPC-induced 数据）
  -> 固定 Model-A replay + C1 训练
  -> development benchmark
  -> Round 2 hard-case pool + 选择连续 branch
  -> C2 训练
  -> 锁定 final benchmark（Direct IK / A / C1 / C2）
  -> 少量 threaded_asap 墙钟实时验证
```

Round 1 **不是**只采集 `multi_joint_sine`、`waypoint`、`chirp`。正式配额为 800 个 episode：320 multi-joint sine、240 waypoint、120 chirp、80 circle、40 figure-8。`ellipse` 和 `square` 不进入采集池，只留作最终 OOD 测试。

## 0. 环境、路径和 GPU 前置检查

```bash
conda activate pendulum-rl
export MODEL_A=dynamics_modeling/outputs/checkpoints/gru_20260717_152930
export MODEL_A_DATA='dynamics_modeling/outputs/datasets/irb2400_parallel_data copy.npz'
export DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
python -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'torch_cuda': torch.version.cuda})"
```

正式的 800 episode 采集和训练应当显示 `cuda_available: True`。若为 `False`，只能用于脚本冒烟测试；不要在 CPU 上启动正式 Round 1。需要先安装与驱动匹配的 CUDA 版 PyTorch，再重新执行检查。可额外运行 `nvidia-smi` 确认 GPU 空闲。

下面所有大规模采集都使用 `virtual_asap`：它不执行墙钟 `sleep`，但仍逐个推进 10 ms 的逻辑控制步、保留 packet 延迟和旧计划执行。`threaded_asap` 仅用于最后的实时性验证。

## 1. 标定逻辑 anticipation delay

在目标 GPU 上测 1,000 次规划。输出文件会记录 p95、guard 和最终 `anticipation_delay_steps`；之后 R1、R2 和 benchmark 都必须复用它，不能手填旧机器的 delay。

```bash
python scripts/model_c/calibrate_delay.py \
  --checkpoint "$MODEL_A/best_model.pt" --normalizer "$MODEL_A/normalizer.pt" \
  --device "$DEVICE" --plans 1000 --output_path outputs/model_c/timing.json

python -c "import json; d=json.load(open('outputs/model_c/timing.json')); print({k:d[k] for k in ('planning_time_p95_s','guard_s','anticipation_delay_steps')})"
```

可在这之后进行一个独立的短冒烟运行；该目录不能与正式 Round 1 混用：

```bash
python scripts/model_c/collect_data.py \
  --output_dir outputs/model_c/smoke --episodes 1 \
  --trajectory_counts multi_joint_sine:1 --max_execution_steps 40 \
  --timing_json outputs/model_c/timing.json \
  --checkpoint "$MODEL_A/best_model.pt" --normalizer "$MODEL_A/normalizer.pt" \
  --device "$DEVICE"
```

## 2. Round 1：用 Model A 采集 MPC-induced 数据

```bash
python scripts/model_c/collect_data.py \
  --output_dir outputs/model_c/round1 \
  --timing_json outputs/model_c/timing.json \
  --checkpoint "$MODEL_A/best_model.pt" --normalizer "$MODEL_A/normalizer.pt" \
  --device "$DEVICE"
```

不要加 `--max_execution_steps`；它仅用于冒烟测试。采集器强制使用 GRU history=16、horizon=25、128 CEM samples、2 CEM iterations、每 5 步 replan、500 个执行步和 `virtual_asap`。

每个主 episode 在目标时刻 0、50、…、450 后，首次成功激活的 packet 上尝试一次 branch group。branch 从真实 activation state 出发，开环执行已投影的 25 步 `q_ref`；它的含义是 `open_loop_counterfactual_branch`，不是实际 ASAP 执行轨迹。每条 branch 保存 15 步仅作历史 context 加 25 步有效 transition，因而每条完整 branch 有 6 个合法的 20-step GRU rollout 窗口。

完成后确认 `outputs/model_c/round1/manifest.json`、`transitions_*.npz` 与 `branches_*.npz` 均存在。主轨迹和其全部 branch 共用 `split_group_id`，训练/验证不会跨 parent episode 泄漏。实际有效窗口数应由后续训练输出统计决定，不能把理论上限当成完成条件。

## 3. 构建固定 Model-A replay 和 C1 训练集

先一次性按完整 episode 从 Model-A 原始数据构造固定 replay。脚本顺序解压到临时 memmap，不会把原始大 NPZ 整体读入内存；完成后默认清理 staging 文件。

```bash
python scripts/model_c/build_replay.py \
  --source_path "$MODEL_A_DATA" \
  --output_path outputs/model_c/model_a_replay_1m.npz

python scripts/model_c/build_dataset.py \
  --input outputs/model_c/model_a_replay_1m.npz \
  --input outputs/model_c/round1 \
  --output_path outputs/model_c/C1_train.npz
```

此处 source id 为 Model A = 0、Round 1 = 1。`C1_train.npz` 只保留训练字段，候选诊断仍留在 shard/sidecar。检查每个输出旁的 manifest，尤其是实际 transition 数、有效窗口数和 source 统计。

## 4. 训练 C1（冻结 Model-A normalizer）

```bash
python dynamics_modeling/scripts/train_dynamics.py \
  --data_path outputs/model_c/C1_train.npz \
  --model_type gru --history_len 16 --target_mode delta_dq --control_dt 0.01 \
  --batch_size 4096 --epochs 30 --lr 1e-5 \
  --rollout_loss_steps 20 --rollout_loss_weight 0.025 \
  --init_from_checkpoint "$MODEL_A/best_model.pt" \
  --normalizer_path "$MODEL_A/normalizer.pt" --freeze_normalizer \
  --source_weights 0:0.5,1:0.5 --save_dir outputs/model_c/checkpoints
```

训练结束时记下打印的运行目录，下面记为 `<C1_RUN>`，例如：

```bash
ls -dt outputs/model_c/checkpoints/gru_* | head -1
```

必须检查该运行目录中的 normalizer 哈希与 Model A 相同。这里使用 `--init_from_checkpoint`，而不是 `--resume_checkpoint`：新阶段不应继承 Model A 的 optimizer 状态。带来源权重的 sampler 会重复较小来源，因此“epoch”不是传统完整遍历；请记录训练输出中的 `samples_per_epoch`、`optimizer_steps_total`、每来源覆盖率和重采样倍数。

## 5. 先跑 development benchmark

development 用于选择 C1 checkpoint 和决定 Round 2 的覆盖重点；final 集在 C2 完成前不得运行。先产生两套不同 seed 的不可变 reference，再写入 manifest。

```bash
export D=$(python -c 'import json; print(json.load(open("outputs/model_c/timing.json"))["anticipation_delay_steps"])')

python scripts/model_c/generate_benchmark_references.py \
  --output_dir outputs/model_c/benchmark_references/development --delay "$D" --seed 20260720
python scripts/model_c/generate_benchmark_references.py \
  --output_dir outputs/model_c/benchmark_references/final --delay "$D" --seed 20260721

python scripts/model_c/build_benchmarks.py --kind development \
  --output_path outputs/model_c/benchmarks/development.json \
  --task_reference_dir outputs/model_c/benchmark_references/development --delay "$D"
python scripts/model_c/build_benchmarks.py --kind final \
  --output_path outputs/model_c/benchmarks/final.json \
  --task_reference_dir outputs/model_c/benchmark_references/final --delay "$D" \
  --disjoint_from outputs/model_c/benchmarks/development.json
```

如果 final manifest 构建时提示 reference overlap，说明使用的是修复前生成的 final reference。仅重建 final reference（不要重建 development，也不要删除 development manifest）：

```bash
python scripts/model_c/generate_benchmark_references.py \
  --output_dir outputs/model_c/benchmark_references/final --delay "$D" --seed 20260721 --overwrite
python scripts/model_c/build_benchmarks.py --kind final \
  --output_path outputs/model_c/benchmarks/final.json \
  --task_reference_dir outputs/model_c/benchmark_references/final --delay "$D" \
  --disjoint_from outputs/model_c/benchmarks/development.json
```

每份 manifest 默认有 70 个固定 case：multi-sine、waypoint、chirp、circle、figure-8 各 10 个 ID case，ellipse、square 各 10 个 OOD case。下面只运行 development：

```bash
python scripts/model_c/evaluate.py \
  --manifest outputs/model_c/benchmarks/development.json --device "$DEVICE" \
  --save_dir outputs/model_c/eval/development \
  --model_spec DirectIK,direct_ik \
  --model_spec A,"$MODEL_A/best_model.pt","$MODEL_A/normalizer.pt",gru \
  --model_spec C1,outputs/model_c/checkpoints/<C1_RUN>/best_model.pt,"$MODEL_A/normalizer.pt",gru
```

记录 failure rate、closed-loop tracking RMSE、25-step replay error 和按 case 配对的 bootstrap 区间。若 C1 有多个 checkpoint，使用 development 的预先定义接受条件选出一个，并把它固定为 `<C1_RUN>`。

## 6. Round 2：C1 的 hard-case 定向数据

先用 C1 采一个宽覆盖 pool，再选择连续 branch，而不是抽取离散 transition。采集器会同时保存两类 branch：`strict_exact_action` 只用于诊断 prediction anchor 与真实 activation command 的错位；`activation_projected` 从真实 activation command/velocity 出发逐步满足关节、速度和加速度约束，且模型预测与 MuJoCo 分支使用同一条投影后动作序列。只有后者进入 C2。

若曾用旧版本写入 `outputs/model_c/round2_pool`，保留它作为 strict 不可执行率日志，但不要用于 C2；旧 sidecar 没有 activation-projected branch。以下使用新目录 `round2_v2`。600 episode 配额是 800 episode R1 的 75%，目标为 15,000 个、每个 40-row 的 projected branch。

```bash
python scripts/model_c/collect_data.py \
  --output_dir outputs/model_c/round2_v2 --episodes 600 \
  --trajectory_counts multi_joint_sine:240,waypoint:180,chirp:90,circle:60,figure8:30 \
  --timing_json outputs/model_c/timing.json \
  --checkpoint outputs/model_c/checkpoints/<C1_RUN>/best_model.pt \
  --normalizer "$MODEL_A/normalizer.pt" --device "$DEVICE" --round_name round2

# 15,000 个 40-row branch 约为 60 万条原始行。
python scripts/model_c/select_round2_cases.py --input_dir outputs/model_c/round2_v2 \
  --target_branches 15000 --output_path outputs/model_c/round2_selection.json
python -c "import json; d=json.load(open('outputs/model_c/round2_selection.json')); print(d.get('selection_counts', {}), len(d.get('selected_branch_ids', [])))"
```

选择标签是非互斥的：high model/replay error (35%)、high tracking error (25%)、ranking flip (20%)、residual saturation (10%)、recovery/fallback neighborhood (10%)。若稀有类别不足，脚本会按优先级补足并写入 selection manifest；`late-drop` 仅作为调度日志，不是专门过采样的动力学类别。

候选排序诊断应分开保存，不能将 strict 和 projected 指标混合：

```bash
python scripts/model_c/analyze_candidate_branches.py --input_dir outputs/model_c/round1 \
  --branch_kind strict_exact_action \
  --output_path outputs/model_c/round1_strict_candidate_metrics.json
python scripts/model_c/analyze_candidate_branches.py --input_dir outputs/model_c/round2_v2 \
  --branch_kind strict_exact_action \
  --output_path outputs/model_c/round2_strict_candidate_metrics.json
python scripts/model_c/analyze_candidate_branches.py --input_dir outputs/model_c/round2_v2 \
  --branch_kind activation_projected \
  --output_path outputs/model_c/round2_projected_candidate_metrics.json
```

## 7. 构建并训练 C2

```bash
python scripts/model_c/build_dataset.py \
  --input outputs/model_c/model_a_replay_1m.npz \
  --input outputs/model_c/round1 \
  --input outputs/model_c/round2_v2 \
  --selection_manifest outputs/model_c/round2_selection.json \
  --output_path outputs/model_c/C2_train.npz

python dynamics_modeling/scripts/train_dynamics.py \
  --data_path outputs/model_c/C2_train.npz \
  --model_type gru --history_len 16 --target_mode delta_dq --control_dt 0.01 \
  --batch_size 4096 --epochs 20 --lr 5e-6 \
  --rollout_loss_steps 20 --rollout_loss_weight 0.025 \
  --init_from_checkpoint outputs/model_c/checkpoints/<C1_RUN>/best_model.pt \
  --normalizer_path "$MODEL_A/normalizer.pt" --freeze_normalizer \
  --source_weights 0:0.35,1:0.25,2:0.40 --save_dir outputs/model_c/checkpoints
```

记下生成的 `<C2_RUN>`。C2 的 source id 依次为 Model A=0、R1=1、R2=2。再次确认 normalizer 由文件复制而来且哈希不变，并确认 `split_group_id` 的训练/验证切分报告没有 parent episode 泄漏。

## 8. 运行 final benchmark 与验收

只有 C2 checkpoint、训练配置和选择策略全部固定后，才创建 marker 并运行 final。此操作使 final manifest 可读；不要在 C2 调参期间执行。

```bash
touch outputs/model_c/benchmarks/C2_COMPLETE

python scripts/model_c/evaluate.py \
  --manifest outputs/model_c/benchmarks/final.json --allow_final_benchmark \
  --device "$DEVICE" --save_dir outputs/model_c/eval/final \
  --candidate_metrics A,outputs/model_c/round1_strict_candidate_metrics.json \
  --candidate_metrics C1,outputs/model_c/round2_projected_candidate_metrics.json \
  --model_spec DirectIK,direct_ik \
  --model_spec A,"$MODEL_A/best_model.pt","$MODEL_A/normalizer.pt",gru \
  --model_spec C1,outputs/model_c/checkpoints/<C1_RUN>/best_model.pt,"$MODEL_A/normalizer.pt",gru \
  --model_spec C2,outputs/model_c/checkpoints/<C2_RUN>/best_model.pt,"$MODEL_A/normalizer.pt",gru
```

查看 `outputs/model_c/eval/final/model_abc_summary.csv`、`paired_bootstrap.json` 和候选指标输出。报告至少包含 Direct IK、A、C1、C2；候选 branch 还应报告 predicted/realized cost 的 Spearman 相关、selected-vs-baseline/alternative 的排序准确率、selection regret、anchor prediction error，以及 k=1/5/10/20/25 rollout RMSE。

建议接受 C1/C2 的条件为：相对前一模型 25-step replay error 至少下降 10%、闭环 tracking RMSE 改善、failure rate 不上升、任一 ID 类别不恶化超过 5%，且 OOD 不出现明显崩溃。最终结论以 case-paired bootstrap 区间为准，不能只依据单一的加权总分。

## 9. 最后的墙钟实时性验证

大规模数据已全部由 `virtual_asap` 获得。最后只选少量固定 benchmark case，以相同 checkpoint 和同一 delay 运行 `threaded_asap`，记录 planning time、packet late-drop、fallback 与实际 100 Hz 控制节拍。不要把这一步得到的数据再混入 C2 训练集。

## 10. MuJoCo Oracle-MPC 上限实验

Oracle-MPC 用与 learned MPC 完全相同的 residual 参数化、horizon、CEM 随机样本、迭代次数、cost、约束、reference、seed 和逻辑 delay，只把 learned dynamics rollout 换成从同一完整状态反复恢复的 MuJoCo clone rollout。delay anchor 本身也由 clone 推进 `D` 步得到。

这是离线理论上限，不是实时控制器。Oracle 的墙钟计算通常超过 `D × 10 ms`，但仍固定在 launch 后第 `D` 个逻辑控制步激活；输出会保留实际 `planning_time`、`replan_deadline_miss` 和 `oracle_overrun_scheduled`。因此不能把它的 planning time 或 update rate写成实时可部署结果。

先用两个 case 做冒烟测试。这里使用新目录，不会覆盖已经完成的 final 结果；`--resume` 会校验 reference、checkpoint、normalizer 和全部运行参数的 fingerprint，完全一致时才复用已完成 rollout。

```bash
python scripts/model_c/evaluate.py \
  --manifest outputs/model_c/benchmarks/final.json --allow_final_benchmark \
  --case_ids multi_joint_sine_00,circle_00 --resume \
  --device "$DEVICE" --save_dir outputs/model_c/eval/oracle_smoke \
  --model_spec Oracle,oracle
```

确认两个 case 均没有 planner failure，并检查：

```bash
python -c "import numpy as np; p='outputs/model_c/eval/oracle_smoke/multi_joint_sine_00/Oracle/rollout.npz'; d=np.load(p); print(d['dynamics_backend'], d['oracle_fixed_logical_delay'], np.unique(d['packet_event'], return_counts=True))"
```

随后运行五种控制器的完整比较。因为原来的 final 已经揭盲，本实验是 post-hoc dynamics upper-bound diagnostic；不得再用 Oracle 结果重新选择 C1/C2 checkpoint。Direct IK 只在有任务空间 IK reference 的 circle、figure-8、ellipse 和 square case 上定义，A/C1/C2/Oracle 则运行全部 70 个 case。

```bash
python scripts/model_c/evaluate.py \
  --manifest outputs/model_c/benchmarks/final.json --allow_final_benchmark --resume \
  --device "$DEVICE" --save_dir outputs/model_c/eval/final_with_oracle \
  --model_spec DirectIK,direct_ik \
  --model_spec A,"$MODEL_A/best_model.pt","$MODEL_A/normalizer.pt",gru \
  --model_spec C1,outputs/model_c/checkpoints/<C1_RUN>/best_model.pt,"$MODEL_A/normalizer.pt",gru \
  --model_spec C2,outputs/model_c/checkpoints/<C2_RUN>/best_model.pt,"$MODEL_A/normalizer.pt",gru \
  --model_spec Oracle,oracle
```

该命令会重新运行五种控制器，保证在同一个输出目录内得到可配对的汇总。耗时很长时可以直接中断，之后原命令加 `--resume` 继续。每完成一个 case/model，`model_abc_summary.csv` 都会更新；最终配对区间写入 `paired_bootstrap.json`。

主 Oracle 固定使用 manifest 中的 `128 × 2` CEM 预算。如果主 Oracle 仍不好，再运行高预算诊断；`OracleHighBudget,oracle,512,4` 只覆盖 samples 和 iterations，其他参数继续使用相同 manifest：

```bash
python scripts/model_c/evaluate.py \
  --manifest outputs/model_c/benchmarks/final.json --allow_final_benchmark --resume \
  --device "$DEVICE" --save_dir outputs/model_c/eval/oracle_budget_diagnostic \
  --model_spec Oracle,oracle \
  --model_spec OracleHighBudget,oracle,512,4
```

建议先用 `--case_ids` 选择主 Oracle 表现较差的固定 case；只有确有必要时才运行全部 70 个高预算 case。结果解释如下：

- Oracle 显著优于 A，而 C1/C2 没改善：dynamics 仍是瓶颈，当前聚合数据或训练目标没有解决它。
- Oracle 只比 A 好一点：Model A 已接近当前 MPC 配置的模型上限，Model C 提升空间有限。
- Oracle rollout 准确但闭环 tracking 仍差：优先检查 cost、nominal、residual bound、delay、控制频率和约束，而不是继续训练 C3。
- OracleHighBudget 明显优于 128×2 Oracle：主要瓶颈是 CEM optimizer 预算，而不是 learned dynamics。

论文中应同时报告 tracking/failure 指标和 Oracle 实际 planning time/deadline-miss，并明确 Oracle 使用固定逻辑 delay，不具备实时性。

## 常见边界

- 不要对 `collect_model_c_data.py` 用 shell 循环和 `--append`；collector 自身会写不可变 shard 与 manifest。
- 不要更换或重拟合 C1/C2 normalizer；必须一直使用 Model A 的 `normalizer.pt`。
- 不要删除 reference 的 horizon + delay lookahead padding；500 步只是执行段长度。
- strict branch 不能重投影；它是原计划的反事实诊断。activation-projected branch 也不开反馈和重规划，但会复现部署层的逐步命令约束投影，并以投影后的动作作为预测与训练目标。
- `C2_COMPLETE` 一旦创建，final benchmark 应视为已揭盲，之后不再据其结果调参。
