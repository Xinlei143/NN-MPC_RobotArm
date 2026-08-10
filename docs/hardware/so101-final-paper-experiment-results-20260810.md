# SO101 最终论文实机实验记录

> 这是实验记录和可复现性文档，不是论文正文。正式结果由 `scripts/analyze_so101_paper_trials.py` 根据原始 NPZ 自动更新。

生成时间（UTC）：2026-08-10T05:38:20.288984+00:00
协议：`so101_final_paper_20260810_tracking1deg_v1`

## 1. 冻结方法

- 最终 NN-MPC：tracking-dominant objective `J = C_q`，`w_q=1`，其余 soft cost 权重均为 0。
- residual authority：每个关节 `|r_j| <= 1°`。保留 Direct、Fixed Preview6、CEM best/mean/fixed-preview 候选池。
- 控制频率：30 Hz；`H=6`；CEM `samples=128`，`iterations=2`；GRU history=16。
- 保留 canonical executable projector、速度/加速度/joint hard constraints、braking、encoder quantization、startup/homing gate。
- `preview6` 固定为 6 steps（200 ms），正式测试不按轨迹重新搜索 preview 长度。

## 2. 实验矩阵

| family | shapes | speeds | matched phase blocks | controllers/block | runs |
|---|---|---|---:|---:|---:|
| held-out | 3 | 2 | 3 | 3 | 54 |
| development reference | 1 | 1 | 3 | 3 | 9 |
| total |  |  |  |  | 63 |

控制器顺序按 repeat 预注册为：repeat 0 `Direct → Preview6 → NN-MPC`；repeat 1 `Preview6 → NN-MPC → Direct`；repeat 2 `NN-MPC → Direct → Preview6`。

## 3. 指标定义

主评估窗口只取冻结 reference 的 `SEGMENT_SHAPE_LOOP`；不把 startup、approach、return、padding 混入主 RMSE。
- joint RMSE / P95 / max：编码器 `q_ctrl` 相对 frozen `q_des`，单位 degree。
- TCP 指标：编码器关节角经 fine MuJoCo model FK 得到的位置误差，单位 mm；没有外部定位仪，因此不称为外部实测 TCP。
- command activity：transmitted executable command 的 velocity/acceleration RMS、P95、max；residual 同时报告 requested 与 executed。`requested residual` 是 CEM/planner 请求的 residual，在 executable-command projector 之前受 ±1° authority 约束；`executed deviation` 是最终 transmitted command 相对该 tick instantaneous nominal reference 的偏差，经过 stateful velocity/acceleration projection、braking 和 encoder quantization 后计算，因此可以超过 1°。
- safety/timing：deadline miss、planner failure/late drop、command v/a violation、TX failure、projection flags、planner latency。编码器量化造成的离散命令加速度超限单独报告，不计入 runtime safety violation；技术不完整 trial 不进入 controller aggregate，但保留在 ledger。

## 4. 开发阶段参考结果（非正式 held-out 统计）

| controller | circle joint RMSE | 说明 |
|---|---:|---|
| Direct IK | 0.9019° | development circle，preview 0 |
| Fixed Preview6 | 0.6482° | development circle，固定 6 steps |
| NN-MPC ±1° | 0.4024° | tracking-only，development circle |

这些数值用于记录方法冻结前的开发证据，不能替代下面的 paired formal matrix。此前 ±2° authority 结果只作为 authority ablation，不属于最终方法。

## 5. 正式结果（自动汇总）

| family | shape | speed | controller | n | joint RMSE (deg) | TCP RMSE (mm) | command vel RMS (rad/s) | command acc RMS (rad/s²) | runtime v/a violations | requested residual P95 / max (deg) | executed deviation P95 / max (deg) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| circle | circle | development | direct | 3 | 0.9893 | 9.557 | 0.0559 | 0.8069 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| circle | circle | development | nn_mpc | 3 | 0.3605 | 3.765 | 0.0851 | 1.1872 | 0.0 / 0.0 | 1.0000 / 1.0000 | 1.1886 / 1.9125 |
| circle | circle | development | preview6 | 3 | 0.6300 | 6.958 | 0.0559 | 0.8069 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| heldout | back_and_forth | fast | direct | 3 | 0.9476 | 9.398 | 0.0487 | 0.8054 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| heldout | back_and_forth | fast | nn_mpc | 3 | 0.3734 | 3.817 | 0.0787 | 1.1842 | 0.0 / 0.0 | 1.0000 / 1.0000 | 1.0963 / 1.8288 |
| heldout | back_and_forth | fast | preview6 | 3 | 0.6589 | 6.949 | 0.0487 | 0.8054 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| heldout | back_and_forth | nominal | direct | 3 | 0.8761 | 8.594 | 0.0379 | 0.7656 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| heldout | back_and_forth | nominal | nn_mpc | 3 | 0.3286 | 3.411 | 0.0773 | 1.1937 | 0.0 / 0.0 | 1.0000 / 1.0000 | 1.2290 / 1.8991 |
| heldout | back_and_forth | nominal | preview6 | 3 | 0.6434 | 6.857 | 0.0379 | 0.7656 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| heldout | ellipse | fast | direct | 3 | 0.8772 | 9.403 | 0.0472 | 0.8001 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| heldout | ellipse | fast | nn_mpc | 3 | 0.3457 | 3.812 | 0.0840 | 1.2073 | 0.0 / 0.0 | 1.0000 / 1.0000 | 1.2183 / 1.8705 |
| heldout | ellipse | fast | preview6 | 3 | 0.6335 | 6.419 | 0.0472 | 0.8003 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| heldout | ellipse | nominal | direct | 3 | 0.8340 | 8.534 | 0.0376 | 0.8013 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| heldout | ellipse | nominal | nn_mpc | 3 | 0.3167 | 3.591 | 0.0786 | 1.2054 | 0.0 / 0.0 | 1.0000 / 1.0000 | 1.2689 / 1.9755 |
| heldout | ellipse | nominal | preview6 | 3 | 0.6298 | 6.545 | 0.0376 | 0.8015 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| heldout | rounded_square | fast | direct | 3 | 1.0048 | 9.060 | 0.0504 | 0.7329 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| heldout | rounded_square | fast | nn_mpc | 3 | 0.4168 | 3.935 | 0.0876 | 1.2087 | 0.0 / 0.0 | 1.0000 / 1.0000 | 1.1868 / 1.8894 |
| heldout | rounded_square | fast | preview6 | 3 | 0.7731 | 6.028 | 0.0504 | 0.7333 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| heldout | rounded_square | nominal | direct | 3 | 0.9182 | 8.485 | 0.0394 | 0.7143 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| heldout | rounded_square | nominal | nn_mpc | 3 | 0.3376 | 3.725 | 0.0782 | 1.1916 | 0.0 / 0.0 | 1.0000 / 1.0000 | 1.1878 / 1.8971 |
| heldout | rounded_square | nominal | preview6 | 3 | 0.7093 | 6.429 | 0.0394 | 0.7146 | 0.0 / 0.0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

Paired差值（NN-MPC − baseline；负值表示 NN-MPC 更低）：

- `nn_mpc_minus_direct:tracking_joint_rmse_deg`：n=18，mean=-0.55654，std=0.0572867
- `nn_mpc_minus_direct:tracking_tcp_rmse_mm`：n=18，mean=-5.19736，std=0.732082
- `nn_mpc_minus_preview6:tracking_joint_rmse_deg`：n=18，mean=-0.32152，std=0.0566019
- `nn_mpc_minus_preview6:tracking_tcp_rmse_mm`：n=18，mean=-2.82296，std=0.877266
- 主表 paired 统计只使用 18 个 held-out matched pairs；包含 development circle 的 21-pair 统计保留在 `analysis/trial_metrics.json` 的 `paired` 字段中。

## 6. 结果分析（自动生成）

### 6.1 Held-out 总体性能

正式矩阵共完成 63/63 个 trial，其中 held-out 集合为 54 个 trial（3 种轨迹 × 2 种速度 × 3 个 phase × 3 个控制器）。所有 trial 均达到预期步数，因此没有因执行不完整而被排除。

| controller | held-out joint RMSE (deg) | held-out TCP RMSE (mm) | command velocity RMS (rad/s) | command acceleration RMS (rad/s²) | requested residual P95 / max (deg) | executed deviation P95 / max (deg) |
|---|---:|---:|---:|---:|---:|---:|
| Direct IK | 0.910 | 8.912 | 0.0435 | 0.7699 | 0.000 / 0.000 | 0.000 / 0.000 |
| Fixed Preview6 | 0.675 | 6.538 | 0.0435 | 0.7701 | 0.000 / 0.000 | 0.000 / 0.000 |
| NN-MPC ±1° | **0.353** | **3.715** | 0.0807 | 1.1985 | 1.000 / 1.000 | 1.198 / 1.893 |

在 held-out 集合上，NN-MPC 相对 Direct 将 joint RMSE 降低 61.2%，将 FK-derived TCP RMSE 降低 58.3%；相对 Fixed Preview6 仍分别降低 47.7% 和 43.2%。因此结果支持一个有边界的结论：learned state-dependent correction 的收益不只是固定 6-step preview 的重复。

### 6.2 Paired consistency、速度和轨迹形状

held-out 中有 18 个 matched phase-condition pairs。NN-MPC 的 joint RMSE 相对 Direct 的差值 18/18 为负（mean -0.5565°，std 0.0573°，95% CI [-0.5850, -0.5281]°）；相对 Preview6 的差值同样为 18/18（mean -0.3215°，std 0.0566°，95% CI [-0.3497, -0.2934]°）。TCP 的对应 paired mean 差值为 -5.197 mm（95% CI [-5.561, -4.833]，18/18 wins）和 -2.823 mm（95% CI [-3.259, -2.387]，18/18 wins）。

| held-out speed | Direct joint / TCP | Preview6 joint / TCP | NN-MPC joint / TCP |
|---|---:|---:|---:|
| nominal | 0.876° / 8.538 mm | 0.661° / 6.610 mm | **0.328° / 3.575 mm** |
| fast | 0.943° / 9.287 mm | 0.688° / 6.466 mm | **0.379° / 3.855 mm** |

从 nominal 到 fast，Direct 的 held-out joint RMSE 从 0.876° 增至 0.943°；NN-MPC 从 0.328° 增至 0.379°，但仍保持低于两个 baseline。三种 held-out 形状和两档速度中，NN-MPC 的平均 RMSE 均低于 Direct 和 Preview6；最困难的 rounded-square/fast 条件下，NN-MPC 仍达到约 0.417° joint RMSE，而 Direct 和 Preview6 分别约为 1.005° 和 0.773°。

### 6.3 实时性与安全性

所有 63 个 trial 的 runtime safety violation、planner failure、control deadline miss 均为 0。21 个 NN-MPC trial 共记录 20956 个 planner events，global planner latency mean/P95/P99/max 为 22.73/28.36/29.93/68.09 ms；其中 3 个结果为 `success_late_dropped`（3/20956），但没有引起 control deadline miss 或 trial technical exclusion。P95 低于 33.3 ms 控制周期，但这里将 planner latency 与 control-period deadline 分开报告。

日志中的 `quantized accel exceedance` 不计入安全违规。它来自 30 Hz 下离散编码器计数计算出的 command acceleration，属于诊断性量化效应；真实运行时安全字段均为 0。

### 6.4 Tracking–command activity trade-off

held-out 集合中，NN-MPC 的 command velocity RMS 为 0.0807 rad/s，相比 Direct 的 0.0435 rad/s 增加约 85.4%；command acceleration RMS 为 1.1985 rad/s²，相比 Direct 的 0.7699 rad/s² 增加约 55.7%。NN-MPC 的 requested residual P95/max 为 1.000/1.000°，executed deviation P95/max 的逐 trial 平均为 1.198/1.893°；所有 held-out NN-MPC raw rollout 的 requested max 为 1.000°，executed deviation 全局 max 为 2.118°。因此性能提升伴随更高的 command activity；当前数据支持 tracking accuracy 与 command smoothness 之间存在明确 trade-off，但不支持“同时降低 tracking error 和 command activity”的更强结论。

### 6.5 结论与解释边界

综合 54 个 held-out trial 和 9 个 circle development trial，结果支持 tracking-dominant NN-MPC 在固定 ±1° residual authority 下具有稳定的实机收益。该结论限定于当前 SO101 硬件、冻结 GRU checkpoint、30 Hz 控制周期、H=6、CEM 128×2 和本协议中的参考轨迹；不能外推为对其他机器人、模型或轨迹分布的普遍保证。TCP 数值来自编码器关节角的 MuJoCo FK，不是外部定位仪测量。


## 7. Trial ledger

| trial | controller | shape | speed | phase | repeat | status | joint RMSE (deg) | TCP RMSE (mm) | safety violations | quantized accel exceedance |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|
| circle_p0_direct | direct | circle | development | 0 | 0 | complete | 1.0147 | 9.605 | 0 | 798 |
| circle_p0_preview6 | preview6 | circle | development | 0 | 0 | complete | 0.6720 | 6.608 | 0 | 775 |
| circle_p0_nn_mpc | nn_mpc | circle | development | 0 | 0 | complete | 0.3872 | 3.876 | 0 | 1006 |
| circle_p2_preview6 | preview6 | circle | development | 1 | 1 | complete | 0.6511 | 7.865 | 0 | 805 |
| circle_p2_nn_mpc | nn_mpc | circle | development | 1 | 1 | complete | 0.3322 | 3.522 | 0 | 1007 |
| circle_p2_direct | direct | circle | development | 1 | 1 | complete | 0.9967 | 10.619 | 0 | 786 |
| circle_p4_nn_mpc | nn_mpc | circle | development | 2 | 2 | complete | 0.3621 | 3.898 | 0 | 1005 |
| circle_p4_direct | direct | circle | development | 2 | 2 | complete | 0.9566 | 8.446 | 0 | 751 |
| circle_p4_preview6 | preview6 | circle | development | 2 | 2 | complete | 0.5670 | 6.400 | 0 | 751 |
| ellipse_nominal_p0_direct | direct | ellipse | nominal | 0 | 0 | complete | 0.8204 | 8.504 | 0 | 890 |
| ellipse_nominal_p0_preview6 | preview6 | ellipse | nominal | 0 | 0 | complete | 0.6308 | 6.218 | 0 | 890 |
| ellipse_nominal_p0_nn_mpc | nn_mpc | ellipse | nominal | 0 | 0 | complete | 0.3085 | 3.594 | 0 | 1159 |
| ellipse_nominal_p1_preview6 | preview6 | ellipse | nominal | 1 | 1 | complete | 0.6365 | 6.779 | 0 | 857 |
| ellipse_nominal_p1_nn_mpc | nn_mpc | ellipse | nominal | 1 | 1 | complete | 0.3148 | 3.365 | 0 | 1159 |
| ellipse_nominal_p1_direct | direct | ellipse | nominal | 1 | 1 | complete | 0.8272 | 8.671 | 0 | 857 |
| ellipse_nominal_p2_nn_mpc | nn_mpc | ellipse | nominal | 2 | 2 | complete | 0.3268 | 3.812 | 0 | 1157 |
| ellipse_nominal_p2_direct | direct | ellipse | nominal | 2 | 2 | complete | 0.8544 | 8.429 | 0 | 889 |
| ellipse_nominal_p2_preview6 | preview6 | ellipse | nominal | 2 | 2 | complete | 0.6220 | 6.640 | 0 | 889 |
| ellipse_fast_p0_direct | direct | ellipse | fast | 0 | 0 | complete | 0.8986 | 9.309 | 0 | 716 |
| ellipse_fast_p0_preview6 | preview6 | ellipse | fast | 0 | 0 | complete | 0.6616 | 6.981 | 0 | 716 |
| ellipse_fast_p0_nn_mpc | nn_mpc | ellipse | fast | 0 | 0 | complete | 0.3348 | 3.839 | 0 | 934 |
| ellipse_fast_p1_preview6 | preview6 | ellipse | fast | 1 | 1 | complete | 0.6407 | 6.186 | 0 | 677 |
| ellipse_fast_p1_nn_mpc | nn_mpc | ellipse | fast | 1 | 1 | complete | 0.3343 | 3.706 | 0 | 932 |
| ellipse_fast_p1_direct | direct | ellipse | fast | 1 | 1 | complete | 0.8798 | 9.941 | 0 | 677 |
| ellipse_fast_p2_nn_mpc | nn_mpc | ellipse | fast | 2 | 2 | complete | 0.3678 | 3.889 | 0 | 931 |
| ellipse_fast_p2_direct | direct | ellipse | fast | 2 | 2 | complete | 0.8533 | 8.960 | 0 | 714 |
| ellipse_fast_p2_preview6 | preview6 | ellipse | fast | 2 | 2 | complete | 0.5980 | 6.090 | 0 | 714 |
| back_and_forth_nominal_p0_direct | direct | back_and_forth | nominal | 0 | 0 | complete | 0.9090 | 8.978 | 0 | 714 |
| back_and_forth_nominal_p0_preview6 | preview6 | back_and_forth | nominal | 0 | 0 | complete | 0.6782 | 6.731 | 0 | 714 |
| back_and_forth_nominal_p0_nn_mpc | nn_mpc | back_and_forth | nominal | 0 | 0 | complete | 0.3366 | 3.646 | 0 | 1102 |
| back_and_forth_nominal_p1_preview6 | preview6 | back_and_forth | nominal | 1 | 1 | complete | 0.5843 | 5.681 | 0 | 803 |
| back_and_forth_nominal_p1_nn_mpc | nn_mpc | back_and_forth | nominal | 1 | 1 | complete | 0.2963 | 2.953 | 0 | 1093 |
| back_and_forth_nominal_p1_direct | direct | back_and_forth | nominal | 1 | 1 | complete | 0.8179 | 7.812 | 0 | 803 |
| back_and_forth_nominal_p2_nn_mpc | nn_mpc | back_and_forth | nominal | 2 | 2 | complete | 0.3530 | 3.634 | 0 | 1098 |
| back_and_forth_nominal_p2_direct | direct | back_and_forth | nominal | 2 | 2 | complete | 0.9014 | 8.993 | 0 | 800 |
| back_and_forth_nominal_p2_preview6 | preview6 | back_and_forth | nominal | 2 | 2 | complete | 0.6678 | 8.159 | 0 | 800 |
| back_and_forth_fast_p0_direct | direct | back_and_forth | fast | 0 | 0 | complete | 1.0491 | 10.195 | 0 | 567 |
| back_and_forth_fast_p0_preview6 | preview6 | back_and_forth | fast | 0 | 0 | complete | 0.7327 | 7.073 | 0 | 563 |
| back_and_forth_fast_p0_nn_mpc | nn_mpc | back_and_forth | fast | 0 | 0 | complete | 0.3329 | 3.378 | 0 | 879 |
| back_and_forth_fast_p1_preview6 | preview6 | back_and_forth | fast | 1 | 1 | complete | 0.5785 | 5.511 | 0 | 671 |
| back_and_forth_fast_p1_nn_mpc | nn_mpc | back_and_forth | fast | 1 | 1 | complete | 0.3969 | 4.268 | 0 | 875 |
| back_and_forth_fast_p1_direct | direct | back_and_forth | fast | 1 | 1 | complete | 0.8905 | 8.436 | 0 | 671 |
| back_and_forth_fast_p2_nn_mpc | nn_mpc | back_and_forth | fast | 2 | 2 | complete | 0.3905 | 3.805 | 0 | 874 |
| back_and_forth_fast_p2_direct | direct | back_and_forth | fast | 2 | 2 | complete | 0.9032 | 9.562 | 0 | 653 |
| back_and_forth_fast_p2_preview6 | preview6 | back_and_forth | fast | 2 | 2 | complete | 0.6656 | 8.264 | 0 | 653 |
| rounded_square_nominal_p0_direct | direct | rounded_square | nominal | 0 | 0 | complete | 0.9092 | 9.098 | 0 | 752 |
| rounded_square_nominal_p0_preview6 | preview6 | rounded_square | nominal | 0 | 0 | complete | 0.7010 | 6.680 | 0 | 739 |
| rounded_square_nominal_p0_nn_mpc | nn_mpc | rounded_square | nominal | 0 | 0 | complete | 0.2949 | 3.283 | 0 | 1062 |
| rounded_square_nominal_p1_preview6 | preview6 | rounded_square | nominal | 1 | 1 | complete | 0.7026 | 6.576 | 0 | 739 |
| rounded_square_nominal_p1_nn_mpc | nn_mpc | rounded_square | nominal | 1 | 1 | complete | 0.3459 | 3.611 | 0 | 1064 |
| rounded_square_nominal_p1_direct | direct | rounded_square | nominal | 1 | 1 | complete | 0.9078 | 8.503 | 0 | 739 |
| rounded_square_nominal_p2_nn_mpc | nn_mpc | rounded_square | nominal | 2 | 2 | complete | 0.3721 | 4.280 | 0 | 1066 |
| rounded_square_nominal_p2_direct | direct | rounded_square | nominal | 2 | 2 | complete | 0.9378 | 7.852 | 0 | 739 |
| rounded_square_nominal_p2_preview6 | preview6 | rounded_square | nominal | 2 | 2 | complete | 0.7241 | 6.030 | 0 | 739 |
| rounded_square_fast_p0_direct | direct | rounded_square | fast | 0 | 0 | complete | 0.9637 | 9.316 | 0 | 587 |
| rounded_square_fast_p0_preview6 | preview6 | rounded_square | fast | 0 | 0 | complete | 0.7587 | 5.997 | 0 | 587 |
| rounded_square_fast_p0_nn_mpc | nn_mpc | rounded_square | fast | 0 | 0 | complete | 0.4367 | 4.131 | 0 | 857 |
| rounded_square_fast_p1_preview6 | preview6 | rounded_square | fast | 1 | 1 | complete | 0.7916 | 5.938 | 0 | 587 |
| rounded_square_fast_p1_nn_mpc | nn_mpc | rounded_square | fast | 1 | 1 | complete | 0.3956 | 3.774 | 0 | 859 |
| rounded_square_fast_p1_direct | direct | rounded_square | fast | 1 | 1 | complete | 1.0350 | 9.200 | 0 | 587 |
| rounded_square_fast_p2_nn_mpc | nn_mpc | rounded_square | fast | 2 | 2 | complete | 0.4181 | 3.900 | 0 | 858 |
| rounded_square_fast_p2_direct | direct | rounded_square | fast | 2 | 2 | complete | 1.0158 | 8.664 | 0 | 587 |
| rounded_square_fast_p2_preview6 | preview6 | rounded_square | fast | 2 | 2 | complete | 0.7689 | 6.150 | 0 | 587 |

## 8. 身份与原始数据

- held-out reference manifest SHA-256：`785e2f924e55077287300b3313105df732fe27c92cd3d0e256e8908db652b1dc`
- raw output root：`outputs/hardware/so101_paper_final/20260810_tracking1deg_v1`
- hardware config：`configs/hardware/so101_follower.local.yaml`
- checkpoint：`dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809/gru_20260809_150038/best_rollout_model.pt`
- normalizer：`dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809/gru_20260809_150038/normalizer.pt`
- 每个 trial 目录应包含 `trial_manifest.json`、`trial.log`、`rollout.npz`；不得覆盖非空 trial，重试必须使用 retry index。

## 9. 解释边界

- 只要 trial 技术上完整，就算 tracking 差也保留，不以结果好坏排除。
- FK-derived TCP 是编码器状态的模型换算，用于辅助报告；没有外部仪器，不作绝对 TCP 测量声明。
- 任何正式方法参数变更都需要新 protocol id 和新输出根目录，不能覆盖本协议。
