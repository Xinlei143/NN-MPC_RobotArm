# GRU Validation

The frozen checkpoint was evaluated on 20 held-out MuJoCo rollouts of 200
steps. Ten used action standard deviation 0.5 and ten used 0.8.

| Action std. | 1-step q RMSE | 20-step q RMSE | Divergence at 20 steps |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.00337 rad | 0.02164 rad | 0 |
| 0.8 | 0.00715 rad | 0.04411 rad | 0.02 |

Teacher-forcing one-step q RMSE was 0.00084 rad at 0.5 and 0.00128 rad at
0.8. The model is suitable for short-horizon prediction within the sampled
training distribution; this validation does not establish globally stable
long-horizon prediction.
