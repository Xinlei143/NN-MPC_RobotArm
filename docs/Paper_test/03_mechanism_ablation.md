# Mechanism Ablation

The matched matrix contains four trajectories, five CEM seeds, and four
variants. Positive values below mean that removing the component worsened TCP
RMSE.

| Removed component | TCP-RMSE change | 95% bootstrap CI |
| --- | ---: | ---: |
| Future alignment | +2.82 mm | [1.26, 4.41] mm |
| Execution-time reanchoring | +654.60 mm | [526.19, 776.58] mm |
| Fast feedback | -0.95 mm | [-2.26, 0.29] mm |

The evidence supports future alignment and, most strongly, execution-time
reanchoring for an absolute-reference interface. The nominal ablation does not
show an independent tracking gain from fast feedback; it remains an execution
correction mechanism rather than a proven primary source of accuracy.
