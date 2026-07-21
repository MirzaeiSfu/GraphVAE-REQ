# Aggregated GraphVAE results
Values are the mean ± sample standard deviation across the completed training seeds shown. Each seed contributes its final test-set mean; the standard deviation here is across training seeds, not across repeated GNN evaluations within one seed. Higher F1-PR, precision, and recall are better; lower MMD values are better. Final test evaluation used each run's saved best-validation checkpoint.

Rows are grouped by dataset and ordered with the baseline first so that all completed variants can be compared directly. PROTEINS' baseline source directory is named `00_graphvae`; it is labeled Setting 01 here. Alpha applies to both `alpha_motif_loss` and `alpha_syntactic_literal_motif_loss`. `F1-PR only` means the checkpoint was selected exclusively by validation F1-PR; all other rows retain their original `table3_priority` selector.

## Combined primary and third-party metrics

| Dataset | Experiment group | Variant | Alpha | Selector | Seeds | F1-PR ↑ | MMD RBF ↓ | Precision ↑ | Recall ↑ | 3rd-party F1-PR ↑ | 3rd-party MMD RBF ↓ |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| AIDS | Previous | 01 baseline | 0 | table3_priority | 3 | 0.8854 ± 0.0442 | 0.00007 ± 0.00003 | 0.8337 ± 0.0605 | 0.9477 ± 0.0286 | 0.9107 ± 0.0311 | 0.00016 ± 0.00003 |
| AIDS | New hyperparameter | 03 top-150 | 0.01 | table3_priority | 3 | 0.8665 ± 0.0163 | 0.00148 ± 0.00228 | 0.8170 ± 0.0340 | 0.9262 ± 0.0196 | 0.8959 ± 0.0087 | 0.00158 ± 0.00228 |
| AIDS | New hyperparameter | 03 previous rules | 0.05 | table3_priority | 3 | 0.8979 ± 0.0127 | 0.00229 ± 0.00204 | 0.8475 ± 0.0294 | 0.9564 ± 0.0173 | 0.9161 ± 0.0087 | 0.00236 ± 0.00203 |
| AIDS | Previous | 03 motif | 0.1 | table3_priority | 3 | 0.8920 ± 0.0209 | 0.00351 ± 0.00195 | 0.8394 ± 0.0248 | 0.9525 ± 0.0151 | 0.9045 ± 0.0187 | 0.00343 ± 0.00167 |
| ENZYMES | Previous | 01 baseline | 0 | table3_priority | 3 | 0.9046 ± 0.0210 | 0.02118 ± 0.03190 | 0.8647 ± 0.0367 | 0.9503 ± 0.0089 | 0.9133 ± 0.0167 | 0.01826 ± 0.02592 |
| ENZYMES | Previous | 03 top-150 | 0.01 | table3_priority | 3 | 0.9440 ± 0.0093 | 0.02392 ± 0.03019 | 0.9142 ± 0.0161 | 0.9767 ± 0.0186 | 0.9083 ± 0.0318 | 0.02671 ± 0.03441 |
| ENZYMES | Previous | 03 motif | 0.1 | table3_priority | 3 | 0.9206 ± 0.0502 | 0.00084 ± 0.00047 | 0.9047 ± 0.0658 | 0.9386 ± 0.0423 | 0.9214 ± 0.0171 | 0.00134 ± 0.00018 |
| ENZYMES | F1-PR validation | 03 top-100 | 0.1 | F1-PR only | 3 | 0.9355 ± 0.0341 | 0.07222 ± 0.03344 | 0.9353 ± 0.0520 | 0.9381 ± 0.0425 | 0.9202 ± 0.0121 | 0.06417 ± 0.02779 |
| MUTAG | Previous | 01 baseline | 0 | table3_priority | 3 | 0.7378 ± 0.0206 | 0.12313 ± 0.03588 | 0.6017 ± 0.0296 | 1.0000 ± 0.0000 | 0.6443 ± 0.0262 | 0.14838 ± 0.03597 |
| MUTAG | Previous | 03 motif | 0.1 | table3_priority | 3 | 0.7799 ± 0.0604 | 0.09185 ± 0.04115 | 0.6530 ± 0.0843 | 0.9991 ± 0.0015 | 0.6929 ± 0.0476 | 0.10587 ± 0.04685 |
| MUTAG | F1-PR validation | 03 motif | 0.1 | F1-PR only | 3 | 0.8385 ± 0.0107 | 0.08223 ± 0.01006 | 0.7308 ± 0.0179 | 1.0000 ± 0.0000 | 0.7470 ± 0.0328 | 0.08836 ± 0.01208 |
| OGBG-MOLBBBP | Previous | 01 baseline | 0 | table3_priority | 3 | 0.7387 ± 0.0478 | 0.05202 ± 0.08276 | 0.6370 ± 0.0602 | 0.8855 ± 0.0354 | 0.7571 ± 0.0396 | 0.06656 ± 0.08521 |
| OGBG-MOLBBBP | New hyperparameter | 03 previous rules | 0.01 | table3_priority | 3 | 0.7187 ± 0.0125 | 0.10708 ± 0.04634 | 0.6673 ± 0.0379 | 0.7860 ± 0.0284 | 0.7691 ± 0.0281 | 0.14320 ± 0.03947 |
| OGBG-MOLBBBP | New hyperparameter | 03 previous rules | 0.03 | table3_priority | 3 | 0.7396 ± 0.0696 | 0.00382 ± 0.00291 | 0.6574 ± 0.0693 | 0.8519 ± 0.0743 | 0.7705 ± 0.0597 | 0.01098 ± 0.00061 |
| OGBG-MOLBBBP | Previous | 03 motif | 0.1 | table3_priority | 3 | 0.7757 ± 0.0258 | 0.05422 ± 0.09020 | 0.7105 ± 0.0099 | 0.8588 ± 0.0523 | 0.8007 ± 0.0103 | 0.06538 ± 0.10100 |
| OGBG-MOLBBBP | Previous | 06 motif + temperature | 0.1 | table3_priority | 2 | 0.7726 ± 0.0590 | 0.05832 ± 0.08012 | 0.7048 ± 0.0750 | 0.8588 ± 0.0325 | 0.7880 ± 0.0572 | 0.10184 ± 0.13181 |
| PROTEINS | Previous | 01 baseline | 0 | table3_priority | 3 | 0.8915 ± 0.0433 | 0.07392 ± 0.01674 | 0.8949 ± 0.0485 | 0.8902 ± 0.0508 | 0.9005 ± 0.0269 | 0.09283 ± 0.01652 |
| PROTEINS | New hyperparameter | 03 previous rules | 0.01 | table3_priority | 3 | 0.8848 ± 0.0373 | 0.07808 ± 0.01500 | 0.8896 ± 0.0615 | 0.8833 ± 0.0407 | 0.8745 ± 0.0084 | 0.09126 ± 0.02220 |
| PROTEINS | New hyperparameter | 03 previous rules | 0.03 | table3_priority | 3 | 0.9306 ± 0.0269 | 0.06659 ± 0.01626 | 0.9029 ± 0.0363 | 0.9616 ± 0.0340 | 0.9229 ± 0.0294 | 0.07314 ± 0.00292 |
| PROTEINS | Previous | 03 motif | 0.1 | table3_priority | 3 | 0.9291 ± 0.0237 | 0.02603 ± 0.04391 | 0.8995 ± 0.0466 | 0.9633 ± 0.0129 | 0.9199 ± 0.0393 | 0.01895 ± 0.03165 |
| PROTEINS | F1-PR validation | 03 motif | 0.1 | F1-PR only | 3 | 0.9029 ± 0.0099 | 0.00270 ± 0.00206 | 0.8711 ± 0.0397 | 0.9405 ± 0.0265 | 0.9149 ± 0.0316 | 0.00322 ± 0.00326 |
| PROTEINS | Previous | 06 motif + temperature | 0.1 | table3_priority | 3 | 0.9271 ± 0.0308 | 0.02605 ± 0.04389 | 0.8971 ± 0.0520 | 0.9616 ± 0.0154 | 0.9197 ± 0.0381 | 0.01898 ± 0.03162 |
| PTC | Previous | 01 baseline | 0 | table3_priority | 3 | 0.8182 ± 0.0392 | 0.09410 ± 0.03032 | 0.7219 ± 0.0407 | 0.9557 ± 0.0374 | 0.8204 ± 0.0256 | 0.10507 ± 0.01383 |
| PTC | Previous | 03 motif | 0.1 | table3_priority | 3 | 0.8724 ± 0.0400 | 0.06885 ± 0.03625 | 0.8062 ± 0.0508 | 0.9538 ± 0.0222 | 0.8757 ± 0.0433 | 0.07629 ± 0.03183 |

## Combined structural metrics

| Dataset | Experiment group | Variant | Alpha | Selector | Seeds | Degree ↓ | Clustering ↓ | Orbit ↓ | Spectral ↓ | Diameter ↓ | Generated edges | Reference edges |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIDS | Previous | 01 baseline | 0 | table3_priority | 3 | 0.00569 ± 0.00077 | 0.29487 ± 0.02418 | 0.00390 ± 0.00036 | 0.00733 ± 0.00116 | 0.05013 ± 0.00986 | 36.96 ± 6.46 | 16.17 ± 0.79 |
| AIDS | New hyperparameter | 03 top-150 | 0.01 | table3_priority | 3 | 0.00475 ± 0.00028 | 0.22950 ± 0.01677 | 0.00355 ± 0.00115 | 0.00672 ± 0.00084 | 0.04684 ± 0.00866 | 101.52 ± 101.70 | 16.17 ± 0.79 |
| AIDS | New hyperparameter | 03 previous rules | 0.05 | table3_priority | 3 | 0.00294 ± 0.00087 | 0.19143 ± 0.00764 | 0.00321 ± 0.00178 | 0.00565 ± 0.00065 | 0.03912 ± 0.00634 | 144.04 ± 89.87 | 16.17 ± 0.79 |
| AIDS | Previous | 03 motif | 0.1 | table3_priority | 3 | 0.00284 ± 0.00114 | 0.18032 ± 0.02201 | 0.00359 ± 0.00113 | 0.00573 ± 0.00062 | 0.02862 ± 0.01542 | 190.96 ± 48.58 | 16.17 ± 0.79 |
| ENZYMES | Previous | 01 baseline | 0 | table3_priority | 3 | 0.03393 ± 0.00471 | 0.02235 ± 0.00216 | 0.01886 ± 0.01072 | 0.01443 ± 0.00283 | 0.04965 ± 0.02789 | 148.44 ± 112.80 | 60.66 ± 2.99 |
| ENZYMES | Previous | 03 top-150 | 0.01 | table3_priority | 3 | 0.03382 ± 0.00797 | 0.02323 ± 0.00194 | 0.04836 ± 0.02354 | 0.01752 ± 0.00308 | 0.05062 ± 0.01929 | 202.48 ± 161.66 | 60.66 ± 2.99 |
| ENZYMES | Previous | 03 motif | 0.1 | table3_priority | 3 | 0.03072 ± 0.00433 | 0.02303 ± 0.00207 | 0.01676 ± 0.00545 | 0.01375 ± 0.00098 | 0.01887 ± 0.00376 | 134.00 ± 42.05 | 60.66 ± 2.99 |
| ENZYMES | F1-PR validation | 03 top-100 | 0.1 | F1-PR only | 3 | 0.02630 ± 0.00730 | 0.02198 ± 0.00328 | 0.01632 ± 0.01039 | 0.01225 ± 0.00031 | 0.02474 ± 0.01793 | 49.69 ± 7.03 | 60.66 ± 2.99 |
| MUTAG | Previous | 01 baseline | 0 | table3_priority | 3 | 0.00934 ± 0.00252 | 0.15306 ± 0.08708 | 0.00794 ± 0.00269 | 0.02033 ± 0.00180 | 0.01660 ± 0.01358 | 21.27 ± 0.34 | 18.49 ± 0.44 |
| MUTAG | Previous | 03 motif | 0.1 | table3_priority | 3 | 0.00765 ± 0.00195 | 0.11734 ± 0.04219 | 0.00229 ± 0.00088 | 0.02234 ± 0.00391 | 0.02437 ± 0.00807 | 20.93 ± 3.65 | 18.49 ± 0.44 |
| MUTAG | F1-PR validation | 03 motif | 0.1 | F1-PR only | 3 | 0.00569 ± 0.00288 | 0.07137 ± 0.01288 | 0.00139 ± 0.00021 | 0.02089 ± 0.00466 | 0.01750 ± 0.00936 | 18.67 ± 1.12 | 18.49 ± 0.44 |
| OGBG-MOLBBBP | Previous | 01 baseline | 0 | table3_priority | 3 | 0.01400 ± 0.00143 | 0.22540 ± 0.04671 | 0.00224 ± 0.00172 | 0.01946 ± 0.00299 | 0.14394 ± 0.03307 | 21.71 ± 5.01 | 24.04 ± 0.26 |
| OGBG-MOLBBBP | New hyperparameter | 03 previous rules | 0.01 | table3_priority | 3 | 0.01264 ± 0.00367 | 0.22796 ± 0.01127 | 0.00137 ± 0.00035 | 0.01841 ± 0.00131 | 0.15848 ± 0.03291 | 19.03 ± 1.18 | 24.04 ± 0.26 |
| OGBG-MOLBBBP | New hyperparameter | 03 previous rules | 0.03 | table3_priority | 3 | 0.01370 ± 0.00342 | 0.21453 ± 0.09134 | 0.00289 ± 0.00178 | 0.01800 ± 0.00422 | 0.14735 ± 0.03914 | 27.57 ± 7.33 | 24.04 ± 0.26 |
| OGBG-MOLBBBP | Previous | 03 motif | 0.1 | table3_priority | 3 | 0.01249 ± 0.00201 | 0.18179 ± 0.04217 | 0.00115 ± 0.00074 | 0.02105 ± 0.00165 | 0.18287 ± 0.01263 | 22.54 ± 7.51 | 24.04 ± 0.26 |
| OGBG-MOLBBBP | Previous | 06 motif + temperature | 0.1 | table3_priority | 2 | 0.01116 ± 0.00599 | 0.20070 ± 0.01775 | 0.00107 ± 0.00057 | 0.01722 ± 0.00431 | 0.16044 ± 0.07033 | 22.54 ± 6.07 | 24.18 ± 0.09 |
| PROTEINS | Previous | 01 baseline | 0 | table3_priority | 3 | 0.04650 ± 0.00197 | 0.02961 ± 0.00197 | 0.02660 ± 0.00386 | 0.02379 ± 0.00336 | 0.04200 ± 0.01648 | 41.27 ± 1.51 | 53.48 ± 1.18 |
| PROTEINS | New hyperparameter | 03 previous rules | 0.01 | table3_priority | 3 | 0.04290 ± 0.00382 | 0.02969 ± 0.00308 | 0.01608 ± 0.00125 | 0.02340 ± 0.00509 | 0.03430 ± 0.01428 | 41.70 ± 1.56 | 53.48 ± 1.18 |
| PROTEINS | New hyperparameter | 03 previous rules | 0.03 | table3_priority | 3 | 0.04387 ± 0.00174 | 0.02828 ± 0.00250 | 0.02542 ± 0.00740 | 0.02253 ± 0.00227 | 0.01954 ± 0.01401 | 46.83 ± 2.85 | 53.48 ± 1.18 |
| PROTEINS | Previous | 03 motif | 0.1 | table3_priority | 3 | 0.04280 ± 0.00659 | 0.02899 ± 0.00333 | 0.01265 ± 0.00546 | 0.02309 ± 0.00433 | 0.02726 ± 0.01620 | 103.58 ± 52.72 | 53.48 ± 1.18 |
| PROTEINS | F1-PR validation | 03 motif | 0.1 | F1-PR only | 3 | 0.03874 ± 0.00393 | 0.02603 ± 0.00263 | 0.00700 ± 0.00115 | 0.02103 ± 0.00371 | 0.02471 ± 0.01608 | 134.55 ± 67.24 | 53.48 ± 1.18 |
| PROTEINS | Previous | 06 motif + temperature | 0.1 | table3_priority | 3 | 0.04203 ± 0.00442 | 0.02852 ± 0.00316 | 0.01537 ± 0.01081 | 0.02135 ± 0.00282 | 0.02171 ± 0.01009 | 108.09 ± 56.79 | 53.48 ± 1.18 |
| PTC | Previous | 01 baseline | 0 | table3_priority | 3 | 0.04681 ± 0.00339 | 0.22445 ± 0.08800 | 0.02859 ± 0.01180 | 0.02625 ± 0.00519 | 0.13710 ± 0.04751 | 25.24 ± 1.48 | 27.65 ± 1.88 |
| PTC | Previous | 03 motif | 0.1 | table3_priority | 3 | 0.03172 ± 0.00479 | 0.14198 ± 0.04820 | 0.00447 ± 0.00232 | 0.02107 ± 0.00254 | 0.08632 ± 0.06834 | 21.83 ± 1.96 | 27.65 ± 1.88 |

## Incomplete or externally unavailable runs

These runs are excluded from the combined result tables because a three-seed final-test aggregate is not available. Status snapshot: 2026-07-21.

| Dataset | Variant | Alpha | Accessible final-test seeds | Available result or progress | Status |
|---|---|---:|---:|---|---|
| AIDS | 03 previous rules | 0.01 | 0/3 | Epochs 8570/8052/8592 | All three seeds were preempted; no final test files |
| AIDS | 03 previous rules | 0.03 | 0/3 | Seeds 0/1 stopped at epochs 7174/7420; seed 2 reached epoch 16470 | Seeds 0/1 were preempted; seed 2 was healthy and running when checked |
| PROTEINS | 03 previous rules | 0.05 | 1/3 | Seed 0: F1-PR 0.9558, MMD RBF 0.06908 | Seeds 1/2 artifacts remain on Jie-lab and are not accessible through noninteractive SSH |
| OGBG-MOLBBBP | 03 previous rules | 0.05 | 1/3 | Seed 2: F1-PR 0.7971, MMD RBF 0.00407 | Seeds 0/1 artifacts remain on Jie-lab and are not accessible through noninteractive SSH |

## Comparison cautions

- The `previous rules` alpha sweep is directly comparable because it keeps Setting 03, motif rules, and `table3_priority` selection fixed while changing alpha.
- F1-PR-validation rows change the checkpoint selector in addition to using alpha=0.1.
- AIDS top-150/alpha-0.01 and ENZYMES top-100/top-150 rows use different rule caps, so they are not pure alpha-only comparisons.
- OGBG-MOLBBBP Setting 06 contains only two completed seeds.
