# LOBSTER Setting 6 Exact No-Guard Reproduction

Rows written: 40
CSV: `reports/lobster_setting6_exact_no_guard_reproduction_20260708.csv`

## Key Rows

| Source | Run | Type | Epoch | Score | Edges Gen | MMD RBF | F1-PR |
|---|---|---|---:|---:|---:|---:|---:|
| lobster_docx_20260708 | GraphVAE motif original temp | historical_docx_posthoc |  | 0.465120983015 | 28.7 | 0.1999 | 1 |
| setting6_exact_no_guard | lobster_setting6_exact_no_guard_rep3__cs-cl-09_gpu1 | best_validation_metadata | 8999 | 0.441825136457 | 44.1 | 0.20103699258 | 1.00001 |
| setting6_exact_no_guard | lobster_setting6_exact_no_guard_rep1__cs-cl-19_gpu1 | best_validation_metadata | 2999 | 0.570425944718 | 507.6 | 0.0200214743614 | 0.923298688914 |
| setting6_exact_no_guard | lobster_setting6_exact_no_guard_rep2__cs-cl-26_gpu0 | best_validation_metadata | 2999 | 0.570425944718 | 507.6 | 0.0200214743614 | 0.923298688914 |
| setting6_exact_no_guard | lobster_setting6_exact_no_guard_rep3__cs-cl-09_gpu1 | final_test_eval |  | 1.41979534416 | 527.25 | 0.0305013656616 | 0.831469406375 |
| setting6_exact_no_guard | lobster_setting6_exact_no_guard_rep1__cs-cl-19_gpu1 | final_test_eval |  | 1.63812639088 | 973.95 | 0.0799462080002 | 0.808833903542 |
| setting6_exact_no_guard | lobster_setting6_exact_no_guard_rep2__cs-cl-26_gpu0 | final_test_eval |  | 1.63812639088 | 973.95 | 0.0799462080002 | 0.808833903542 |

## Best Validation Samples By Score

| Run | Epoch | Score | Edges Gen | Degree | Clustering | Orbit | Spectral | Diameter | MMD RBF | F1-PR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lobster_setting6_exact_no_guard_rep3__cs-cl-09_gpu1 | 8999 | 0.441825136457 | 44.1 | 0.0143473233309 | 0.0989724237874 | 0.0388039353841 | 0.0306606716635 | 0.0154190385838 | 0.20103699258 | 1.00001 |
| lobster_setting6_exact_no_guard_rep3__cs-cl-09_gpu1 | 2999 | 0.544767942097 | 54.9 | 0.0193266246948 | 0.0730333805589 | 0.0265239550546 | 0.0363080644735 | 0.032809744311 | 0.144951860514 | 0.947378448753 |
| lobster_setting6_exact_no_guard_rep1__cs-cl-19_gpu1 | 2999 | 0.570425944718 | 507.6 | 0.0287968474075 | 0.219417069797 | 0.0769913476297 | 0.0442982968695 | 0.0784715029841 | 0.0200214743614 | 0.923298688914 |
| lobster_setting6_exact_no_guard_rep2__cs-cl-26_gpu0 | 2999 | 0.570425944718 | 507.6 | 0.0287968474075 | 0.219417069797 | 0.0769913476297 | 0.0442982968695 | 0.0784715029841 | 0.0200214743614 | 0.923298688914 |
| lobster_setting6_exact_no_guard_rep3__cs-cl-09_gpu1 | 6999 | 0.602160110613 | 32.9 | 0.00993632418897 | 0.34414568412 | 0.0216870129867 | 0.0473580547748 | 0.075651773441 | 0.213633323008 | 1.00001 |
| lobster_setting6_exact_no_guard_rep1__cs-cl-19_gpu1 | 4999 | 0.610386750169 | 495.3 | 0.0165431179567 | 0.133055042122 | 0.0537026640825 | 0.0481571296222 | 0.21040645785 | 0.0200624227524 | 0.947378448753 |
| lobster_setting6_exact_no_guard_rep2__cs-cl-26_gpu0 | 4999 | 0.610386750169 | 495.3 | 0.0165431179567 | 0.133055042122 | 0.0537026640825 | 0.0481571296222 | 0.21040645785 | 0.0200624227524 | 0.947378448753 |
| lobster_setting6_exact_no_guard_rep1__cs-cl-19_gpu1 | 6999 | 0.726243412579 | 23.4 | 0.0254476281511 | 0.121867577896 | 0.061047527431 | 0.0446273179933 | 0.0687737047072 | 0.205804370545 | 0.947378448753 |
| lobster_setting6_exact_no_guard_rep2__cs-cl-26_gpu0 | 6999 | 0.726243412579 | 23.4 | 0.0254476281511 | 0.121867577896 | 0.061047527431 | 0.0446273179933 | 0.0687737047072 | 0.205804370545 | 0.947378448753 |
| lobster_setting6_exact_no_guard_rep3__cs-cl-09_gpu1 | 10000 | 0.751128686722 | 21.5 | 0.0375476083781 | 0.0637917164107 | 0.0455040683155 | 0.0598125394993 | 0.175720719232 | 0.21554484684 | 1.00001 |
