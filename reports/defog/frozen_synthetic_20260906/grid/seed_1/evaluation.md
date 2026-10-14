# Legacy DGL Random-GIN evaluation

The input PyG collection was converted directly to DGL before the existing evaluator was invoked.

| Mode | Metric | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| topology_control | f1_pr | 0.0177964 | 0.0533325 | 1e-05 | 0.177794 |
| topology_control | mmd_linear | 6917.14 | 7467.46 | 163.522 | 20981.5 |
| topology_control | mmd_rbf | 0.73698 | 0.156388 | 0.468897 | 0.988949 |
| topology_control | precision | 0.01 | 0.03 | 0 | 0.1 |
| topology_control | recall | 0.79 | 0.302324 | 0 | 1 |
