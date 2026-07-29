# Delay Sweep

The sweep covers circle and fast ellipse, four protocols, delays 2/4/6/8, and
three seeds. Pooled TCP RMSE (mm):

| Protocol | D2 | D4 | D6 | D8 |
| --- | ---: | ---: | ---: | ---: |
| Naive | 54.5 | 54.8 | 69.6 | 91.0 |
| Anchor-only | 30.9 | 29.6 | 831.4 | 941.2 |
| Anchor + reanchor | 30.1 | 31.5 | 29.1 | 32.4 |
| Full | 32.1 | 30.6 | 32.5 | 33.2 |

Future alignment without reanchoring becomes unstable for D >= 6. Reanchoring
is the component that makes activation alignment deployable for absolute joint
references.
