# Projection Strategy

| Variant | Common-D TCP RMSE | Deployed TCP RMSE | Deployed E2E p95 |
| --- | ---: | ---: | ---: |
| Projection off | 39.67 mm | 36.78 mm | 49.54 ms |
| Full compiled | 34.64 mm | 35.15 mm | 50.61 ms |
| Two-stage compiled | 31.21 mm | 31.89 mm | 44.27 ms |

Two-stage compiled projection passed the pre-specified 5% non-inferiority
criterion against full compiled projection, while reducing deployed solve/E2E
p95 by 6.45/6.34 ms. No velocity or acceleration violation occurred.
