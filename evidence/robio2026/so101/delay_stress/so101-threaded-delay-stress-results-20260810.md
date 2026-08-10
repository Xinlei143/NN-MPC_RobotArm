# SO101 ThreadedAsync 实机延迟压力测试结果

> 本文档是 `so101_threaded_delay_stress_20260810_v2` 的实验记录与结果汇总，不替代正式 SO101 论文实验的 63-trial aggregate，也不修改正式输出根目录。该实验只测试已有的 ThreadedAsync（activation-aligned asynchronous residual CEM-MPC），没有加入 `NaiveDelayed` 硬件对照。

记录日期：2026-08-10  
实验平台：SO101 follower，5 个控制关节，30 Hz   
正式协议：`so101_threaded_delay_stress_20260810_v2`  
基础协议：`so101_final_paper_20260810_tracking1deg_v1`

## 1. 一句话结论

在保持 SO101 的 checkpoint、reference、CEM、residual authority、executable projector 和安全限制不变的情况下，人为增加 33 ms planner wall-clock delay，并使用由 shadow calibration 重新计算的 activation anticipation delay `D=4` 后，ThreadedAsync 在 3 个 paired held-out `ellipse_fast` blocks 上全部完成且没有 deadline、planner 或 runtime safety failure；但 joint RMSE 相对无注入延迟条件平均增加 `+0.1009°`（3/3 pairs 变差，paired Student-t 95% CI `[+0.0249°, +0.1769°]`）。

该结果支持“真实 position-controlled hardware 上的 ThreadedAsync 对额外 planner latency 具有可测量但可执行的退化响应”这一有限结论；它不是 `ThreadedAsync` 与 `NaiveDelayed` 的硬件因果 ablation，因此不能单独证明 activation alignment 在真机上的必要性。

## 2. 术语与比较对象

| 术语 | 本文定义 |
|---|---|
| `ThreadedAsync` | 论文中的 activation-aligned asynchronous residual CEM-MPC 执行路径。 |
| `baseline` | 无额外注入延迟，使用正式 SO101 纸面实验的原始 timing calibration 和 OOD envelope。 |
| `delay33` | 在 planner 完成 CEM、executable projection 和 packet post-processing 后、packet publication 前注入确定性的 33 ms wall-clock delay；随后使用独立 shadow 数据重新标定 timing 与 OOD envelope。 |
| `D` | activation anticipation delay，以 30 Hz control step 计。baseline 使用原始 `D=2`；delay33 使用本次校准得到的 `D=4`。 |
| joint RMSE | 由 encoder joint state 与 frozen joint reference 计算的主指标，单位为 degree。 |
| quantization diagnostic | 由 30 Hz encoder/count quantization 离散差分产生的诊断性 acceleration exceedance；它不等同于 executable projector 的 runtime safety violation。 |

## 3. 冻结的控制器与唯一运行时扰动

两种 active condition 均继承基础 SO101 protocol：

- 30 Hz，5 个控制关节，GRU history length 16；
- prediction horizon `H=6`；CEM `128 samples × 2 iterations`；
- residual authority `±1°`；tracking-dominant objective `J=C_q`；
- 相同 dynamics checkpoint、normalizer、hardware configuration、joint reference、startup/home gate 和 formal repeat seeds；
- 相同 compiled two-stage executable projection，以及 joint、velocity、acceleration、braking 和 quantization 相关处理；
- 相同 `ellipse_fast_p0/p1/p2` held-out reference blocks；控制器顺序按 `p0: baseline→delay33`、`p1: delay33→baseline`、`p2: baseline→delay33` 进行 counterbalancing。

唯一的运行时扰动是：

```text
planner search → executable projection → packet post-processing
→ deterministic 33 ms sleep → packet publication
```

packet 仍遵循 canonical activation semantics；若 packet 不能满足校准后的 activation guard，则按原有规则丢弃，nominal command 保持生效。正式运行前，runner 先通过 torque-disabled read-only connection 完成 planner preflight，确认同一 CUDA/CEM worker 达到 `ready` 后才允许 production connection 和运动。

## 4. Shadow calibration 与 gate

### 4.1 v1 结果仅作为历史诊断

初始 v1 shadow calibration 产生了 1,734 个 latency samples，使用 `p99_plus_guard`，late-drop rate 为 `2.768%`。该结果没有满足 active gate（samples 至少 2,000、方法固定为 `p99.5`、late-drop 和 packet-expiry rate 均低于 1%），因此 v1 artifacts 不进入最终 active comparison，也不与 v2 合并。

### 4.2 v2 独立校准结果

v2 使用独立的 D4 shadow pass，并从 measured p99.5 latency 计算 activation delay，而不是手动调节性能参数：

\[
D=\left\lceil\frac{t_{p99.5}+t_{guard}}{\Delta t}\right\rceil
=\left\lceil\frac{100.601\text{ ms}+5\text{ ms}}{33.333\text{ ms}}\right\rceil=4.
\]

| 项目 | v2 结果 | gate |
|---|---:|---:|
| finite latency samples | 2,327 | ≥ 2,000 |
| calibration method | `p99.5` | fixed |
| calibrated latency | 100.601 ms | — |
| control period | 33.333 ms | — |
| guard | 5.0 ms | — |
| anticipation delay | `D=4` steps = 133.3 ms | derived |
| late-drop rate | 0.0% | < 1% |
| packet-expiry rate | 0.0% | < 1% |
| injected planner delay | 33.0 ms | fixed |

v2 OOD envelope 使用 token semantics `q_ctrl,dq_ctrl,transmitted_q_ref`，阈值为 `37397.54395457484`：

| coverage | 结果 | gate |
|---|---:|---:|
| executed history | 100.000% | ≥ 99% |
| selected action | 99.683% | ≥ 99% |
| predicted state | 99.683% | ≥ 99% |

因此只有 v2 calibration/OOD artifacts 被用于 delay33 active runs。

## 5. Active paired matrix

每个 phase 运行一次 baseline 和一次 delay33，共 6 个技术完整 trial、3 个 matched pairs；每个 trial 记录 1,015/1,015 expected steps。

| matched block | baseline joint RMSE (°) | delay33 joint RMSE (°) | delay33 − baseline (°) |
|---|---:|---:|---:|
| `ellipse_fast_p0` | 0.34685 | 0.41896 | +0.07211 |
| `ellipse_fast_p1` | 0.31772 | 0.45074 | +0.13302 |
| `ellipse_fast_p2` | 0.33127 | 0.42876 | +0.09749 |
| **mean** | **0.33195** | **0.43282** | **+0.10087** |

3/3 paired blocks 中 delay33 的 joint RMSE 高于 baseline。paired difference 的样本标准差为 `0.03059°`；95% CI 使用 3 个 matched pairs 的 paired Student-t interval（df=2），为 `[+0.02488°, +0.17687°]`。

## 6. 聚合 tracking、command activity 与 planner timing

其中 tracking/activity 列是 3 个 active runs 的 arithmetic mean；latency 列则汇总每个 condition 的全部 planner events，不把不同 run 或 event 当作可互换的独立样本。

| condition | injected delay | joint RMSE (°) | command vel RMS (rad/s) | command acc RMS (rad/s²) | pooled latency mean (ms) | pooled P95 (ms) | pooled P99 (ms) | pooled max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 ms | 0.33195 | 0.08315 | 1.20526 | 22.56 | 28.33 | 29.37 | 56.62 |
| delay33 | 33 ms | 0.43282 | 0.09701 | 1.29162 | 79.32 | 94.34 | 99.44 | 110.08 |

相对于 baseline，delay33 的平均 joint RMSE 增加约 `30.4%`，command velocity RMS 增加约 `16.7%`，command acceleration RMS 增加约 `7.2%`。这组结果表明额外 planner latency 会带来可量化的 tracking degradation 和更高的 command activity；它没有显示“延迟增加后 tracking 完全不变”。

每个 active run 的 runtime counters 均为零：

- `tx_failure_count = 0`；
- `control_deadline_miss_count = 0`；
- `planner_failure_count = 0`；
- `planner_late_drop_count = 0`；
- `packet_expiration_count = 0`；
- `command_velocity_violation_count = 0`；
- `command_acceleration_violation_count = 0`；
- `safety_violation_count = 0`。

NPZ 中仍能看到 quantization diagnostic flags：各 run 约为 930–934 次。这些 flags 来源于 30 Hz 编码器/count quantization 的离散 acceleration 诊断，不是 executable projector 违反 command limit；因此它们不与上面的零 runtime safety violation 冲突。

## 7. 结果解释与论文 claim boundary

### 7.1 该实验支持的结论

1. 在真实 SO101 position-controlled manipulator 上，ThreadedAsync 可以在人为增加 33 ms planner delay 后继续完成完整的 held-out tracking episode。
2. 使用由 measured latency 得到的 `D=4` 和对应 OOD envelope 后，6 个 active runs 没有 control deadline miss、planner failure、packet expiry 或 runtime safety violation。
3. 额外延迟仍然会损害 tracking：3/3 matched pairs 变差，平均 joint RMSE 增加 `0.1009°`。
4. 这是一项硬件 stress/deployability validation，与 simulation 中 stale-plan/de-anchoring 的机制结果方向一致，但证据层级不同。

### 7.2 该实验不支持的结论

- 不能把这组结果写成 `ThreadedAsync` 相对 `NaiveDelayed` 在硬件上的因果优势，因为本实验没有运行 `NaiveDelayed` hardware controller。
- 不能写成 zero-shot sim-to-real：delay33 使用 hardware-specific recalibration（`D=4`）和独立 OOD envelope。
- 不能把 FK-derived TCP 误写成 external Cartesian measurement；本实验的 primary endpoint 仍是 encoder-space joint RMSE。
- `n=3` 是小规模 paired stress check，不应与正式 54-trial SO101 baseline comparison 的统计强度混为一谈。

## 8. 可直接用于论文或 supplement 的英文结果段落

> We additionally evaluated the physical ThreadedAsync execution path under a deterministic 33-ms planner delay injected immediately before packet publication. All controller, model, reference, projection, and safety parameters were inherited from the frozen SO101 protocol; the delay condition used an independently calibrated anticipation delay of four 30-Hz control steps based on the measured p99.5 latency. Across three counterbalanced held-out ellipse-fast blocks, all six runs completed without control-loop deadline misses, planner failures, packet expirations, or runtime safety violations. The injected-delay condition increased joint RMSE from 0.3319° to 0.4328° on average, with all three paired differences positive (mean +0.1009°, paired Student-t 95% CI [+0.0249°, +0.1769°]). This stress test supports physical deployability and quantifies latency sensitivity of the aligned execution path; because no NaiveDelayed hardware ablation was performed, it does not independently establish the causal necessity of activation alignment on hardware.

## 9. Reproducibility and artifact locations

离线重新生成 active summary（不会连接或驱动机械臂）：

```bash
conda run -n lerobot python scripts/analyze_so101_delay_stress.py \
  --protocol configs/experiments/so101_threaded_delay_stress_20260810_v2.yaml
```

关键 artifacts：

- protocol：`configs/experiments/so101_threaded_delay_stress_20260810_v2.yaml`
- delay calibration：`outputs/hardware/so101_delay_stress/20260810_threaded_asap_v2/delay_calibration_33ms_d4.json`
- OOD envelope：`outputs/hardware/so101_delay_stress/20260810_threaded_asap_v2/ood_envelope_33ms_d4.json`
- merged shadow：`outputs/hardware/so101_delay_stress/20260810_threaded_asap_v2/calibration/merged_shadow.npz`
- OOD tokens：`outputs/hardware/so101_delay_stress/20260810_threaded_asap_v2/calibration/OOD_TOKENS.npz`
- active summary JSON/CSV：`outputs/hardware/so101_delay_stress/20260810_threaded_asap_v2/active_d4/delay_stress_summary.{json,csv}`
- six raw trial directories：`outputs/hardware/so101_delay_stress/20260810_threaded_asap_v2/active_d4/`

关键文件 SHA-256：

```text
configs/experiments/so101_threaded_delay_stress_20260810_v2.yaml
b9bb81e94298284698515b83df29ed6c9fcf2c043f042e47609eb8912d740406

outputs/hardware/so101_delay_stress/20260810_threaded_asap_v2/delay_calibration_33ms_d4.json
d1c5498a665993487b72f98cf530a397a73bb3286020fff232e6d200177905b7

outputs/hardware/so101_delay_stress/20260810_threaded_asap_v2/ood_envelope_33ms_d4.json
64d352cab2494962dc375b9f90da9fc9936bb7702fbd1db04fd5ec570c5f949c

outputs/hardware/so101_delay_stress/20260810_threaded_asap_v2/active_d4/delay_stress_summary.json
6e124d41559e85204193fcac787e6a0daaee8f275f71af557c9d6156529f934f

outputs/hardware/so101_delay_stress/20260810_threaded_asap_v2/active_d4/delay_stress_summary.csv
048d814a01d7b7b39eff0dd32a8f1b368525eba55dd2680fa78007e09a2c7d14

outputs/hardware/so101_delay_stress/20260810_threaded_asap_v2/calibration/merged_shadow.npz
bef0989315976364474c92f3c0290fa6800583245ed39cc5ac67ccf4c03de9d7

outputs/hardware/so101_delay_stress/20260810_threaded_asap_v2/calibration/OOD_TOKENS.npz
6cda1f6197214dbc2fec4c060f4c2565dd8cdd21709e68d6e8599c0b44a2ba42
```

## 10. 最终状态

v2 delay-stress protocol 已完成并通过 calibration/OOD gate；compact protocol、calibration/OOD artifacts、active summary 和本结果记录已纳入 public evidence bundle。不要将 v1 未通过 gate 的 calibration 当作正式结果，也不要把本实验的结论扩展成硬件 `NaiveDelayed` ablation 或 zero-shot transfer。
