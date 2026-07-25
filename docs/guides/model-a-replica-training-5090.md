# Model A 辅助模型训练：RTX 5090 操作手册

本文用于训练不确定性感知模块的三个辅助 GRU。目标是以已锁定的 Model A 为主模型，训练三个**独立初始化**但训练口径完全一致的 replica，用于后续的 ensemble / OOD 安全实验。

> 不要使用 `--init_from_checkpoint` 或 `--resume_checkpoint`。辅助模型必须从不同随机初始化开始训练；主模型 checkpoint 仅用于锁定实验身份和配对 normalizer。

## 1. 本次实验固定口径

| 项目 | 固定值 |
|---|---|
| 主模型目录 | `dynamics_modeling/outputs/checkpoints/gru_20260717_182930` |
| 主模型 checkpoint | `.../best_model.pt` |
| 冻结 normalizer | `.../normalizer.pt` |
| 训练数据 | `dynamics_modeling/outputs/datasets/irb2400_parallel_data copy.npz` |
| 模型 | GRU, `history_len=16`, `target_mode=delta_dq`, `control_dt=0.01` |
| 优化 | 100 epochs, batch size 16384, AdamW, lr `9e-5` |
| 损失 | one-step MSE + 20-step rollout loss, rollout weight `0.025` |
| 主模型训练 seed | 10（只作记录） |
| 辅助模型 seed | 101、211、307 |

三个辅助模型必须共用主模型的 `normalizer.pt`。这使模型分歧处在同一归一化坐标系中；每个输出目录中会保存该文件的受控副本。

## 2. 迁移到 5090 电脑前的准备

将完整仓库迁移到目标电脑，至少确认下列内容存在：

```text
NN-MPC_RobotArm/
├─ dynamics_modeling/outputs/datasets/irb2400_parallel_data copy.npz
├─ dynamics_modeling/outputs/checkpoints/gru_20260717_182930/
│  ├─ best_model.pt
│  ├─ normalizer.pt
│  └─ config.yaml
└─ scripts/train_uncertainty_ensemble.py
```

在仓库根目录运行。以下命令使用 conda 环境名 `pendulum-rl`；若目标电脑环境名不同，将命令中的名字替换掉即可。

```powershell
cd <你的仓库路径>\NN-MPC_RobotArm

conda run -n pendulum-rl python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0)); print('cudnn=', torch.backends.cudnn.version())"
```

预期 `cuda=True`，并显示 RTX 5090。若 `conda` 不在 PowerShell 的 `PATH` 中，可使用完整路径，例如：

```powershell
& 'D:\anaconda\condabin\conda.bat' run --no-capture-output -n pendulum-rl python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

核对三个关键文件：

```powershell
Get-Item 'dynamics_modeling\outputs\datasets\irb2400_parallel_data copy.npz'
Get-Item 'dynamics_modeling\outputs\checkpoints\gru_20260717_182930\best_model.pt'
Get-Item 'dynamics_modeling\outputs\checkpoints\gru_20260717_182930\normalizer.pt'

Get-Content 'dynamics_modeling\outputs\checkpoints\gru_20260717_182930\config.yaml'
```

## 3. 启动三个 replica 训练

脚本会按 `101 -> 211 -> 307` **串行**训练。即使 5090 显存更大，也不要默认并行运行三份完整训练：每份训练都会加载约 11 GB 数据并执行长 rollout，串行更容易保证稳定、可解释和可恢复。

```powershell
conda run --no-capture-output -n pendulum-rl python scripts\train_uncertainty_ensemble.py
```

若 `conda` 不在 PATH：

```powershell
& 'D:\anaconda\condabin\conda.bat' run --no-capture-output -n pendulum-rl python scripts\train_uncertainty_ensemble.py
```

默认输出目录为：

```text
dynamics_modeling/outputs/uncertainty_ensemble_gru_20260717_182930/
├─ training.log
├─ ensemble_manifest.json
├─ seed_101/gru_<timestamp>/
├─ seed_211/gru_<timestamp>/
└─ seed_307/gru_<timestamp>/
```

每个完整的 `gru_<timestamp>` 目录都应含有：

```text
best_model.pt
latest_model.pt
normalizer.pt
normalizer_provenance.yaml
config.yaml
```

## 4. 查看训练状态

实时追踪当前 batch、loss、epoch 和验证结果：

```powershell
Get-Content 'dynamics_modeling\outputs\uncertainty_ensemble_gru_20260717_182930\training.log' -Tail 30 -Wait
```

查看 GPU 利用率：

```powershell
nvidia-smi -l 2
```

检查已完成的 replica：

```powershell
Get-ChildItem 'dynamics_modeling\outputs\uncertainty_ensemble_gru_20260717_182930' -Recurse -Filter best_model.pt |
  Select-Object FullName, Length, LastWriteTime
```

当前运行到哪个 seed：

```powershell
Select-String -Path 'dynamics_modeling\outputs\uncertainty_ensemble_gru_20260717_182930\training.log' -Pattern '^=== seed'
```

## 5. 中断与恢复

前台运行时可按 `Ctrl+C` 停止。每个 epoch 都会写入 `latest_model.pt`，但当前批量脚本的设计是“每个 seed 一次完整训练”。

若某个 seed 中断，推荐单独恢复该 seed，完成后再运行批量脚本的剩余 seed。先找该 seed 最新目录：

```powershell
Get-ChildItem 'dynamics_modeling\outputs\uncertainty_ensemble_gru_20260717_182930\seed_101' -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty FullName
```

假设得到 `<RUN_DIR>`，恢复到 100 epoch：

```powershell
conda run --no-capture-output -n pendulum-rl python dynamics_modeling\scripts\train_dynamics.py `
  --data_path 'dynamics_modeling\outputs\datasets\irb2400_parallel_data copy.npz' `
  --model_type gru --history_len 16 --target_mode delta_dq --control_dt 0.01 `
  --epochs 100 --batch_size 16384 --lr 9e-5 `
  --rollout_loss_steps 20 --rollout_loss_weight 0.025 `
  --normalizer_path 'dynamics_modeling\outputs\checkpoints\gru_20260717_182930\normalizer.pt' `
  --freeze_normalizer --seed 101 `
  --resume_checkpoint '<RUN_DIR>\latest_model.pt'
```

恢复命令不需要再传 `--save_dir`；训练脚本会继续写回 checkpoint 所在目录。恢复完成后，在最终 `ensemble_manifest.json` 中手动补入该 replica 路径和哈希，或重新执行批量脚本时使用 `--reuse_existing`。

## 6. 训练完成后的验证

确认刚训练的 normalizer 与主模型 normalizer 完全一致：

```powershell
$base = Get-FileHash 'dynamics_modeling\outputs\checkpoints\gru_20260717_182930\normalizer.pt' -Algorithm SHA256
$replicas = Get-ChildItem 'dynamics_modeling\outputs\uncertainty_ensemble_gru_20260717_182930' -Recurse -Filter normalizer.pt
$base
$replicas | ForEach-Object { Get-FileHash $_.FullName -Algorithm SHA256 }
```

检查 manifest 是否恰好记录三个 replica：

```powershell
$manifest = Get-Content 'dynamics_modeling\outputs\uncertainty_ensemble_gru_20260717_182930\ensemble_manifest.json' -Raw | ConvertFrom-Json
$manifest.replica_seeds
$manifest.replicas | Select-Object seed, checkpoint, normalizer, checkpoint_sha256
```

再对每个模型执行相同条件下的动力学 rollout 评估。例如 seed 101：

```powershell
conda run --no-capture-output -n pendulum-rl python dynamics_modeling\scripts\eval_dynamics.py `
  --checkpoint '<SEED_101_RUN_DIR>\best_model.pt' `
  --normalizer '<SEED_101_RUN_DIR>\normalizer.pt' `
  --model_type gru --n_joints 6 --history_len 16 `
  --rollout_len 200 --num_rollouts 10 --warmup_steps 50 --action_std 0.95 `
  --horizons 1,5,10,20,50,100,200 `
  --save_dir 'dynamics_modeling\outputs\figures\uncertainty_seed_101'
```

## 7. 训练完成后，不要立刻开启 online gate

下一步应先离线检查：三个辅助模型与 Model A 的分歧是否在 OOD 条件下与真实多步预测误差、约束风险或 tracking-error 增长相关。只有这个相关性成立，才校准新的 3-model 阈值，并测试 `selected-only + H=3 + soft gate`。

当前 online 控制代码原先固定要求 4 个 replica；接入这三个辅助模型前，需要将该校验调整为“至少 2 个 replica”，并在结果元数据中记录实际 replica 数量。
