# SO101 Model-A 扩展采集 runbook(2026-08-08)

10 个 fast-excitation session(session 48–57,source_id=1)一次采完,严格 append 进 v2 canonical。
目标:修复动力学模型命令响应增益 κ≈0(位置控制语料命令-状态相关性 ≈0)。

## 预检与授权状态(已完成)

- [x] 离线 MuJoCo 预检全绿:`outputs/hardware/so101_pre_mpc/20260808_extension_preflight/session_48..57.json`
      ——70 段全过,min predicted clearance 79.6mm(upper_arm@home,离线 strict 阈值),attempt 1 零重采样;
      fast_sine 限速饱和 88–100%,fast_walk |corr(du,ddq)| 0.28–0.32(vs 旧数据 ≤0.09)
- [x] append 可审计例外(方案 B,用户拍板):plant identity 的 table_safety/voltage_* /hardware_config_sha256
      允许差异;collection_plan 仅允许 v1→v2 一个扩展形状;物理键严格。集成测试 5/5 + pytest 8/8
- [x] v2 canonical 副本已建(sha256 与 v1 逐位一致):
      `outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.npz`

## 采集命令(一次采完,~3.5h)

```bash
for i in $(seq 48 57); do
  conda run -n lerobot python dynamics_modeling/scripts/collect_real_workspace_model_a.py \
    --hardware-config configs/hardware/so101_follower.local.yaml \
    --table-safety-config configs/hardware/so101_table_safety.mesh_guard20_runtime10.local.yaml \
    --output outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.npz \
    --session-index $i --source-id 1 --split train \
    --append --enable-motion --operator-supported-shutdown
done
```

- `set -e` 语义:任一 session 失败即停;失败 shard 保留为 `sessions/*.failure.json`,不污染 canonical
- 每 session ~21–22 min(900s 程序 + 启动/预置/回位)
- 每 session 结束时:回 home + guarded shutdown recovery + 扭矩下电,随后 `validate_dataset` + append

## 采集时监控

| 指标 | 要求 | 位置 |
|---|---|---|
| valid/transitions | ≥99% | 每段进度条 |
| clr | ≥ +80mm(runtime 阈值 −10mm) | 每段进度条 |
| Tmax | <50°C | 每段进度条 |

## 每 session 程序(7 段,纯新激励循环,anchor=home)

hold 15s → fast_walk 180s → fast_sine 180s → step_hold 150s → fast_walk 180s → fast_sine 180s → hold 15s

## 采集完成后(离线,由 Claude 执行)

1. `validate_real_dataset.py` 全语料(58 session)
2. 核对 canonical manifest:collection_runs = 58 次、sample_count 增长、plan 记录 v1 来源
3. 重训(完整命令见下;注意参数是 `--data_path` 而非 `--dataset`,train_dynamics.py 的 argparse 里没有 `--dataset`):
   ```bash
   conda run -n lerobot python dynamics_modeling/scripts/train_dynamics.py \
     --data_path outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.npz \
     --robot_config configs/robots/so101.yaml \
     --model_type gru --history_len 16 --target_mode delta_dq --loss_type huber \
     --control_dt 0.03333333333333333 --lr 0.0001 --batch_size 8192 --epochs 200 \
     --rollout_loss_steps 20 --rollout_loss_weight 0.025 \
     --train_sample_stride 2 --val_sample_stride 1 --val_fraction 0.1 \
     --validation_group_ids 38,39,40,41,42 --test_group_ids 43,44,45,46,47 \
     --source_weights 0:1,1:4 \
     --save_dir dynamics_modeling/outputs/checkpoints_real
   ```
   (旧:新 = 1:4 按 source 平衡;新 checkpoint 存新目录,旧 `gru_20260807_130024` 冻结为基线。
   **坑:`--control_dt` 必须传全精度 `0.03333333333333333`**(train_dynamics.py:573 `np.isclose(atol=1e-12, rtol=0)` 与 so101.yaml `expected_control_dt` 逐位比对,`0.033333` 会直接 `ValueError`))
4. **训练后必须补 3-key plant-identity 元数据更新**(否则 checkpoint 过不了运行时校验):v2 canonical manifest 的 `plant_identity` 继承 v1 旧值(`hardware_config_sha256=984e846`、voltage=null),而实机当前是 `4460c456` + 电压 10.5/9.8、12.8/13.5。用 `scripts/patch_checkpoint_voltage_identity.py`(只改 checkpoint/normalizer 元数据 3 个键,权重不动),产出 sidecar `identity_voltage_<YYYYMMDD>.json` 同款记录。新 checkpoint 目录名是训练时间戳,把 `<new_dir>` 换成实际的:
   ```bash
   conda run -n lerobot python scripts/patch_checkpoint_voltage_identity.py \
     --checkpoint dynamics_modeling/outputs/checkpoints_real/<new_dir> \
     --normalizer dynamics_modeling/outputs/checkpoints_real/<new_dir>/normalizer.pt \
     --hardware-config configs/hardware/so101_follower.local.yaml
   ```
   目录模式自动处理 `best_model.pt` / `best_rollout_model.pt` / `latest_model.pt` + `normalizer.pt`,跑完自动跑 `verify_real_artifact_identity` 门禁(通过才写 sidecar)。先 `--dry-run` 看要改哪些文件。
5. normalizer/OOD 包络重标定(新数据分布) → H=6 模型门禁(learned vs persistence)
6. shadow ≥2000 + active 重测(验证 κ 修复后 MPC 是否产生真实修正)

## 备注

- guard15 profile 文件仍损坏(#22,mesh_collision.bundle_sha256 过期)——不影响本次采集(用 guard20_runtime10,
  其 mesh_collision 已指向 scene_table_guard_25mm.xml);修复留待采集后
- v1 语料当年用 guard15 采集,plan B 的 audited 例外覆盖该差异;25mm guard 更保守
- 采集全程 safety 链:workspace 包络投影(0.25 rad/s)+ runtime/planning 双 MuJoCo 校验 + 电压/温度监督
