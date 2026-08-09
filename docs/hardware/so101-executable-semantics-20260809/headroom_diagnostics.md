# SO101 predictive-control headroom diagnostics (2026-08-09)

这份记录对应三个诊断：Direct preview/lead sweep、MPC residual 与参考速度方向关系、counterfactual candidate ranking。它们的目的不是重新训练模型，而是先回答“这个 circle 任务是否存在可利用的 predictive headroom，以及模型能否把候选动作排对”。

## 已实现的工具

- `scripts/run_real_direct_control.py` 增加 `--preview-steps k` 和 `--lead-time-s tau`。两者只改变发送的 command，评价目标仍保存为原始 `q_des[t]`；lead correction 默认限制在 ±2°。
- `scripts/analyze_direct_preview_sweep.py` 汇总多个 Direct/preview/lead 实机日志，输出整体和逐关节 RMSE。
- `scripts/analyze_mpc_residual_alignment.py` 计算 planner-requested residual、实际 transmitted q_ref−q_des 与 forward `dq_des` 的 Pearson 相关、lead slope、同向率和反向率。
- `scripts/benchmark_candidate_ranking.py` 固定相同 state/history，生成 `direct`、`lead:0.05/0.10/0.15/0.20`、`offset:*`、`active` 候选；用模型预测 q-only cost，并可接收同名实机日志，计算 Spearman、pairwise accuracy、top-1 accuracy。

所有候选 command 都先经过 `robot_runtime/executable_command.py` 的 canonical NumPy state machine；模型 history 使用训练等价的 `[x_t,u_t]`，当前 token 的 `u_t` 被候选的第一拍 executable command 覆写。

## 当前已有日志的结果

### 同一 frozen reference 的 baseline 对照

用 `circle_p0/q_des_ctrl.npy` 的前 1087 行作为评价目标，现有日志得到：

| run | q RMSE | shoulder pan | shoulder lift | elbow | wrist flex | wrist roll |
|---|---:|---:|---:|---:|---:|---:|
| Direct formal | 0.8731° | 0.9018° | 1.1320° | 0.9903° | 0.8530° | 0.0941° |
| Active u-q e14 | 0.9310° | 1.3438° | 0.8907° | 1.0968° | 0.7193° | 0.1197° |

这里的 Direct 数字是用同一 frozen `q_des_ctrl.npy` 重新计算的；此前文档中约 0.920° 的 Direct 数字来自另一份 paired baseline，不能和本表混用。

### 实机 Direct preview sweep（同一 frozen reference）

7 条 preview 轨迹均完成 1087 个样本，使用同一启动流程、同一 canonical executable projector 和同一原始 `q_des` 评价目标。结果如下；preview 的含义是 tick `t` 发送 `q_des[t+k]`，因此 `k=6` 对应 200 ms 的显式提前量。

| run | preview | 总 RMSE [deg] | 相对 preview_0 | shoulder pan | shoulder lift | elbow | wrist flex | wrist roll |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| preview_0 | 0 ms | 0.8292 | — | 0.9102 | 0.9447 | 1.0115 | 0.8278 | 0.0941 |
| preview_1 | 33.3 ms | 0.7604 | 8.30% | 0.7407 | 0.9115 | 0.9092 | 0.8221 | 0.0941 |
| preview_2 | 66.7 ms | 0.7016 | 15.38% | 0.6166 | 0.8871 | 0.8176 | 0.7855 | 0.0941 |
| preview_3 | 100 ms | 0.6677 | 19.47% | 0.5496 | 0.9114 | 0.7205 | 0.7542 | 0.0941 |
| preview_4 | 133.3 ms | 0.6393 | 22.90% | 0.5440 | 0.9043 | 0.6228 | 0.7261 | 0.1205 |
| preview_5 | 166.7 ms | 0.6072 | 26.77% | 0.4675 | 0.8851 | 0.5702 | 0.7085 | 0.1205 |
| preview_6 | 200 ms | **0.5872** | **29.19%** | **0.4545** | **0.8688** | **0.5089** | **0.6993** | 0.1205 |

这已经明确证明当前 `circle_p0 + SO101 + 30 Hz` 存在可利用的 predictive-control headroom：同一任务上仅改变命令时序，RMSE 从 `0.8292°` 降到 `0.5872°`。因此 Active MPC 没有改善，不能归因于 Direct 已经达到硬件上限。preview_6 是本次诊断中的最佳已知候选，但它是开环时序基线，不应直接替代带安全门禁的 MPC。

需要注意：preview 越大，发送命令相对原始 `q_des` 的偏移越大，这是有意的 lead authority，不是 projector mismatch；7 条运行都成功完成且没有出现启动/通信失败。完整汇总见 `preview_sweep/preview_sweep.md` 和 `preview_sweep/preview_sweep.json`。

### Active residual 与参考速度

分析文件：`residual_alignment_e14_active/alignment.json`。

| joint | corr(residual, dq_des) | lead slope [s] | 同向率 | 反向率 |
|---|---:|---:|---:|---:|
| shoulder_pan | -0.8111 | -0.0940 | 0.150 | 0.850 |
| shoulder_lift | +0.5219 | +0.0707 | 0.750 | 0.250 |
| elbow_flex | -0.3416 | -0.0303 | 0.335 | 0.665 |
| wrist_flex | +0.4750 | +0.0266 | 0.766 | 0.234 |
| wrist_roll | 无有效运动样本 | — | — | — |

因此 shoulder_pan 的退化不是“没有 residual”，而是 residual 大量与期望运动反向；它在当前轨迹上更像增加 phase lag，而不是做 lead compensation。elbow 也存在相同方向的问题。该结论是离线相关性诊断，不等同于因果证明，但足以优先检查 candidate ranking 和参考时序。

### 初步 candidate-ranking 结果

输出：`candidate_ranking_e14_direct_active/candidate_ranking.json`。

在 6 个 anchor、只有 `direct` 和已有 `active` 两个真实候选的初步对照中：

- 实机 q-only window cost 均值：Direct `2.47199e-4 rad²`，Active `2.60508e-4 rad²`；
- u-q e14 预测 vs 实机：Spearman 均值 `0.0`，pairwise accuracy `0.5`，top-1 accuracy `0.5`；
- e14 离线模型预测的候选均值 cost：Direct `2.67012e-4`，lead 0.05 s `2.44173e-4`，lead 0.10 s `2.24202e-4`，lead 0.15 s `2.16524e-4`，lead 0.20 s `2.09983e-4`，Active `2.37966e-4`。

同一模型对离散 preview 的预测均值为：preview 1 `2.56679e-4`、preview 2 `2.38927e-4`、preview 3 `2.24779e-4`；预测结果保存在 `candidate_ranking_preview_predicted/`，真实 headroom 仍需下面的真机 sweep 验证。

这个结果只说明当前模型倾向预测“更强 lead 会更好”，但已有 Active 实机没有优于 Direct；由于真实候选只有两个，不能据此做最终 checkpoint 选择。正式 benchmark 至少需要 Direct、preview/lead 2–4 个候选各跑一次，最好每个候选重复两次。

### 同一批 preview 实机日志的 candidate ranking

将上面的 7 条同日实机日志接入同一个 benchmark，在 6 个 anchor（200、350、500、650、800、950）、`H=6`、同一 u-q epoch14 模型下比较。模型预测的候选成本和实机窗口成本都随 preview 增大而下降：

| candidate | 模型预测均值 cost [rad²] | 实机均值 cost [rad²] |
|---|---:|---:|
| direct | 2.3044e-4 | 2.3048e-4 |
| preview:1 | 2.0793e-4 | 2.0027e-4 |
| preview:2 | 1.9619e-4 | 1.5518e-4 |
| preview:3 | 1.8741e-4 | 1.4666e-4 |
| preview:4 | 1.8519e-4 | 1.0829e-4 |
| preview:5 | 1.8199e-4 | 9.6719e-5 |
| preview:6 | 1.7849e-4 | **8.5920e-5** |

u-q epoch14 的模型—实机排序指标为：Spearman `0.7136`、pairwise accuracy `0.8145`、top-1 accuracy `0.6667`（6 个 anchor 中 4 个 top-1 正确）。这比此前只有 Direct/Active 两个候选时的 `pairwise=0.5` 信息量大得多，并且说明该模型已经能捕捉到“preview 越强通常越好”的大趋势；但它仍会在局部 anchor 排错，不能把 CEM 的每次细小 cost 差异视为可信。

本次 benchmark 输出：`candidate_ranking_preview_sweep_e14/candidate_ranking.md` 和 `candidate_ranking_preview_sweep_e14/candidate_ranking.json`。因此后续 checkpoint 选择应优先看 candidate-ranking，而不是单独看 scalar κ；κ 只保留为安全/动力学诊断指标。

## 真机 Direct preview sweep 命令

使用 Direct formal 同一份 1087-row frozen reference：

```bash
REF=outputs/hardware/so101_pre_mpc/20260808_refs_6phase/circle_p0/q_des_ctrl.npy
MANIFEST=outputs/hardware/so101_pre_mpc/20260808_refs_6phase/manifest.json
BASE=outputs/hardware/so101_pre_mpc/20260809_headroom/direct_preview

for K in 0 1 2 3 4 5 6; do
  conda run --no-capture-output -n lerobot python scripts/run_real_direct_control.py \
    --hardware-config configs/hardware/so101_follower.local.yaml \
    --reference-mode joint_file \
    --reference-file "$REF" \
    --reference-manifest "$MANIFEST" \
    --preview-steps "$K" \
    --output "$BASE/preview_${K}/rollout.npz" \
    --enable-motion --operator-supported-shutdown
done
```

每次运行前确认 operator 在位、工作空间清空、急停可用；脚本本身不会自动把多个运行并行化。完成后汇总：

```bash
conda run --no-capture-output -n lerobot python scripts/analyze_direct_preview_sweep.py \
  --reference-file "$REF" \
  --run direct=outputs/hardware/so101_pre_mpc/20260808_formal/direct_circle_p0/rollout.npz \
  --run preview_0="$BASE/preview_0/rollout.npz" \
  --run preview_1="$BASE/preview_1/rollout.npz" \
  --run preview_2="$BASE/preview_2/rollout.npz" \
  --run preview_3="$BASE/preview_3/rollout.npz" \
  --run preview_4="$BASE/preview_4/rollout.npz" \
  --run preview_5="$BASE/preview_5/rollout.npz" \
  --run preview_6="$BASE/preview_6/rollout.npz" \
  --output-dir docs/hardware/so101-executable-semantics-20260809/preview_sweep
```

lead sweep 可单独运行，避免和 preview k 混在同一组：

```bash
for TAU in 0.05 0.10 0.15 0.20 0.25 0.30; do
  TAG=${TAU/./p}
  conda run --no-capture-output -n lerobot python scripts/run_real_direct_control.py \
    --hardware-config configs/hardware/so101_follower.local.yaml \
    --reference-mode joint_file --reference-file "$REF" --reference-manifest "$MANIFEST" \
    --lead-time-s "$TAU" --lead-max-deg 2.0 \
    --output "$BASE/lead_${TAG}/rollout.npz" \
    --enable-motion --operator-supported-shutdown
done
```

## residual 方向关系命令

```bash
conda run --no-capture-output -n lerobot python scripts/analyze_mpc_residual_alignment.py \
  --rollout outputs/hardware/so101_pre_mpc/20260809_executable_semantics/u_minus_q_e14_cuda_graph_active/rollout.npz \
  --output-dir docs/hardware/so101-executable-semantics-20260809/residual_alignment_e14_active \
  --velocity-threshold-deg-s 1.0 --residual-threshold-deg 0.05
```

## candidate-ranking 实机接入方式

preview sweep 完成后，可把同名真实日志接入；先只比较现有 Direct/Active 的最小闭环：

```bash
conda run --no-capture-output -n lerobot python scripts/benchmark_candidate_ranking.py \
  --base-rollout outputs/hardware/so101_pre_mpc/20260809_executable_semantics/u_minus_q_e14_cuda_graph_active/rollout.npz \
  --reference-file outputs/hardware/so101_pre_mpc/20260808_refs_6phase/circle_p0/joint_reference_mpc.npz \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --robot-config configs/robots/so101.yaml \
  --model u_minus_q_e14 \
    dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809/gru_20260809_150038/best_rollout_model.pt \
    dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809/gru_20260809_150038/normalizer.pt \
  --run direct=outputs/hardware/so101_pre_mpc/20260808_formal/direct_circle_p0/rollout.npz \
  --run active=outputs/hardware/so101_pre_mpc/20260809_executable_semantics/u_minus_q_e14_cuda_graph_active/rollout.npz \
  --anchors 200,350,500,650,800,950 --horizon 6 \
  --output-dir docs/hardware/so101-executable-semantics-20260809/candidate_ranking_e14_direct_active
```

当 preview/lead 日志齐全时，把 `--run preview_1=...` 等加入同一命令，并把 `--candidate` 列表显式固定为相同名称。checkpoint 的主要选择指标应改为 ranking accuracy；plant identity 仍是硬门禁，κ/sensitivity 只作为安全和动力学诊断，不再单独决定“最好模型”。

candidate benchmark 中离散 preview 使用 `preview:1`、`preview:2` 这样的名称；例如：

```bash
  --candidate direct --candidate preview:1 --candidate preview:2 --candidate preview:3 --candidate active \
  --run direct=outputs/hardware/so101_pre_mpc/20260808_formal/direct_circle_p0/rollout.npz \
  --run preview:1=outputs/hardware/so101_pre_mpc/20260809_headroom/direct_preview/preview_1/rollout.npz \
  --run preview:2=outputs/hardware/so101_pre_mpc/20260809_headroom/direct_preview/preview_2/rollout.npz \
  --run preview:3=outputs/hardware/so101_pre_mpc/20260809_headroom/direct_preview/preview_3/rollout.npz
```
