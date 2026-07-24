# 三辅助模型的不确定性软安全监督器

## 目的与边界

Model A 仍是唯一执行完整 CEM 优化的动力学模型。三个 replica **不直接输出控制命令**；它们只对 Model A 已选中的短期执行轨迹作独立状态预测，并以预测分歧提供在线 uncertainty score。

这避免了把多个动力学预测错误地平均成控制 residual，同时保留 ensemble 在模型失配时的安全价值。

## 在线流程

```text
Model A CEM -> selected residual sequence + Model A predicted trajectory
                         |
                         v
Primary + 3 replicas predict the selected short horizon (H_u)
                         |
                         v
normalized RMS disagreement score
                         |
       +-----------------+--------------------+
       |                 |                    |
   score <= low     low < score < high      score >= high
       |                 |                    |
 residual = 1x    smooth residual scale    inspect concrete risk
                                            |
                         +------------------+------------------+
                         |                                     |
                     no concrete risk                  limit/error/saturation risk
                         |                                     |
          retain at least min_residual_scale       residual = 0; nominal/IK fallback
```

Concrete risk means any of the following on the Model-A selected trajectory:

- predicted joint limit violation inside the configured margin;
- selected residual is near its configured saturation limit;
- predicted next-step tracking error grows beyond the configured ratio of the current error.

An ensemble wall-clock budget timeout remains conservative: it triggers nominal fallback because no complete uncertainty estimate is available.

## Modes

| Mode | Ensemble computation | Command effect |
|---|---|---|
| `off` | none | normal Model-A residual MPC |
| `ensemble_monitor` | selected-only disagreement | records score and risk, does not modify residual |
| `ensemble_soft_gate` | selected-only disagreement | smoothly attenuates residual; hard fallback only for high disagreement plus risk |

`ensemble_gate` is accepted as a deprecated alias for `ensemble_soft_gate`.

## Replica placeholders

No auxiliary checkpoint is hard-coded. After the three seed-trained models have passed the common final H=20 evaluation, provide their six paths when launching MPC:

```powershell
--uncertainty_checkpoints `
  dynamics_modeling/outputs/uncertainty_replicas/seed_101/<run>/best_model.pt `
  dynamics_modeling/outputs/uncertainty_replicas/seed_211/<run>/best_model.pt `
  dynamics_modeling/outputs/uncertainty_replicas/seed_307/<run>/best_model.pt `
--uncertainty_normalizers `
  dynamics_modeling/outputs/uncertainty_replicas/seed_101/<run>/normalizer.pt `
  dynamics_modeling/outputs/uncertainty_replicas/seed_211/<run>/normalizer.pt `
  dynamics_modeling/outputs/uncertainty_replicas/seed_307/<run>/normalizer.pt
```

The loader accepts at least two replicas; this experiment uses Model A plus three replicas, for a four-model ensemble.

## Initial launch protocol

Start with monitoring. It changes planner timing, so compare it against `off` as an operational-cost measurement, not as a zero-overhead baseline.

```powershell
python scripts/run_cem_mpc.py `
  --multirate_mode threaded_asap `
  --checkpoint <MODEL_A_BEST_MODEL> `
  --normalizer <MODEL_A_NORMALIZER> `
  --model_type gru --history_len 16 `
  --uncertainty_mode ensemble_monitor `
  --uncertainty_horizon 3 `
  --uncertainty_budget_ms 3 `
  --uncertainty_checkpoints <REPLICA_101> <REPLICA_211> <REPLICA_307> `
  --uncertainty_normalizers <NORM_101> <NORM_211> <NORM_307>
```

After calibrating on an ID development set and validating OOD detection, enable soft supervision:

```powershell
--uncertainty_mode ensemble_soft_gate `
--uncertainty_low_threshold <CALIBRATED_LOW> `
--uncertainty_high_threshold <CALIBRATED_HIGH> `
--uncertainty_min_residual_scale 0.20
```

The run logs `uncertainty_score`, `uncertainty_residual_scale`, `uncertainty_high_risk_flags`, `uncertainty_gate_flags`, and stage-2 evaluation time. Recalibrate both thresholds after changing replica count, training protocol, normalization, or uncertainty horizon.
