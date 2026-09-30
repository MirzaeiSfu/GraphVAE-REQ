# Legacy DGL Random-GIN evaluation

The input PyG collection was converted directly to DGL before the existing evaluator was invoked.

| Mode | Metric | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| topology_control | f1_pr | 0.725923 | 0.323046 | 1.99987e-05 | 1.00001 |
| topology_control | mmd_linear | 10.9076 | 12.9663 | 0.919858 | 41.6652 |
| topology_control | mmd_rbf | 0.354267 | 0.116614 | 0.225941 | 0.567383 |
| topology_control | precision | 0.67 | 0.302655 | 0.15 | 1 |
| topology_control | recall | 0.88 | 0.296816 | 0 | 1 |
