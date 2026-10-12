# AIDS GraphVAE reconstruction-weight optimization summary

## Short answer

The non-QM9 AIDS experiment is complete. Bayesian optimization found a pair of
nonuniform reconstruction weights that looked better during the short search,
but that pair was worse on average than uniform weights in the predeclared,
matched multi-seed confirmation. The supported AIDS default remains:

```yaml
alpha_node_feat: 1.0
alpha_edge_feat: 1.0
```

This is a negative experimental result, not a failed run. The studies completed
their exact budgets, the comparison answered its stated question, and no test
or held-out graph was accessed.

## What was completed

The AIDS cache contains 1,849 graphs: 1,294 training, 184 validation, and 371
held-out graphs. Only training and validation were used. The cache remained
read-only and retained its exact 73,822,456-byte SHA-256 on the controller and
both TITAN RTX hosts.

The work included a three-GPU timing/reproducibility qualification, a fresh
14-trial search, and a separately frozen six-trial confirmation. The search and
confirmation are both `FROZEN`; all 20 reservations are `COMPLETE`, with no
failed, waiting, running, unreserved, duplicate, or replacement trial. Their
portable snapshots restore without PostgreSQL.

The final test checkpoint passes 117 non-PostgreSQL distributed and
attribute-BO tests plus all 19 tests in a disposable PostgreSQL schema. The
schema was removed and verified absent. Credential, storage-URL, test-access,
and cache-integrity audits pass.

## Objective and determinism

Every reported optimization value is the validation value at exactly:

```text
evaluation.modes.decoded_node_edge.summary.f1_pr.mean
```

It is not a training-loss value and it is not a test value. It evaluates both
decoded node and edge features on all 184 validation graphs and averages five
fixed evaluator repeats. `test_access` stayed false and final evaluation was
skipped during optimization.

For a fixed checkpoint, evaluator seeds, runtime, and hardware, the evaluator
is deterministic. The timing qualification also reproduced the entire uniform
trial exactly on all three GPUs under a fixed training seed. The objective is
not constant across training seeds, however: GraphVAE training deliberately
changes when the training seed changes. That seed sensitivity is why the final
decision used matched pairs at three seeds instead of trusting one search run.

## Results

At the 100-epoch search fidelity, BO selected trial 12:

| Candidate | Node weight | Edge weight | Validation Attr-F1PR |
| --- | ---: | ---: | ---: |
| BO-selected | 1.4240488736 | 2.4689326521 | 0.4390587283 |
| Uniform anchor | 1.0 | 1.0 | 0.3794140017 |

The apparent search advantage was `+0.0596447266`. It was valid for that
100-epoch, seed-0 selection problem, but it was subject to winner selection and
was not evidence of general improvement.

The confirmation trained both candidates for 250 epochs at matched seeds:

| Training seed | BO-selected | Uniform | Selected minus uniform |
| ---: | ---: | ---: | ---: |
| 0 | 0.5833637554 | 0.7182497866 | -0.1348860312 |
| 1 | 0.5874287703 | 0.5763281590 | +0.0111006113 |
| 2 | 0.5722756428 | 0.6419715870 | -0.0696959442 |
| **Mean** | **0.5810227228** | **0.6455165109** | **-0.0644937880** |

The 95% paired interval is `[-0.2461642965, 0.1171767204]`. It crosses zero,
the mean is negative, and two of three pairs favor uniform. The predeclared
decision is therefore `no_improvement`.

## Why the selected weights did not hold up

The main problem was not simply dataset size. With 1,294 training and 184
validation graphs, AIDS is much larger than the earlier LOBSTER cache and has a
meaningful validation split. More data can reduce uncertainty, but it cannot
remove selection bias or a mismatch between search and confirmation fidelity.

Four effects matter most:

1. The winner was the maximum of 14 seed-0 validation observations. Some of
   its apparent advantage is expected to be winner's curse.
2. Search used 100 epochs, while confirmation used 250. Reconstruction weights
   change the optimization trajectory, so candidate rankings need not transfer
   between those checkpoints.
3. The three paired differences vary widely and even change sign. Fixed
   evaluator repeats make measurement stable, but they do not average away
   training-seed variation.
4. Only nine trials were adaptive after five anchors. That is a reasonable
   bounded two-dimensional search, but not enough to map a noisy response
   surface or estimate seed-robust performance.

The selected pair also strengthens edge reconstruction more than node
reconstruction. Those weights are relative to every other GraphVAE loss term,
not merely to each other, so changing their common scale can alter latent and
structural learning. The search optimized the requested proxy correctly; the
proxy was not reliable enough to identify a seed-robust 250-epoch winner.

## Why it took hours

A 100-epoch AIDS trial took about 25--29 minutes. Fourteen trials required five
bounded waves on three GPUs; compute occupied about 2.2 hours and the wall-clock
search, including controller/collection gaps, took about 3.2 hours. A 250-epoch
trial took about 51--57 minutes. The confirmation took about 2.7 hours because
a scheduler defect split its fixed plan into `3+2+1`. That defect is fixed; an
equivalent new six-trial fixed plan should use `3+3` and take roughly 1.8--2.0
hours on the same hardware.

For a new dataset, graph count alone is not a safe runtime predictor. First run
one simultaneous uniform timing trial per GPU at the intended fidelity. A
useful lower-bound estimate is:

```text
search time ~= number of synchronous waves * slowest measured trial time
confirmation time ~= ceil(number of paired trials / workers) * full-fidelity trial time
```

Add time for collection, audits, staging, and any startup barrier. Graph size,
model dimensions, epochs, evaluator graph count, and storage speed all affect
the measured trial time.

## How to use the code for weight optimization

Use the operational guide in
[`attr_f1pr_bayesian_optimization.md`](attr_f1pr_bayesian_optimization.md) and
the AIDS files under `configs/bayesian_optimization/` as a complete example.
The important inputs are:

- a BO-safe YAML such as `aids_graphvae_attr_f1pr_search.yaml`;
- a frozen cache manifest and feature schemas;
- an exact reservation plan and trial count;
- qualified repository, Python, GPU-slot, and protected credential mappings;
- a fresh PostgreSQL study name and fresh artifact root.

Initialize the study with `scripts/run_distributed_graphvae_attr_bo.py`, launch
one bounded synchronous wave at a time, probe ambiguous launches before any
retry, collect only terminal work, audit the exact reservation budget, then
finalize and create a portable snapshot. Select the winner only from validation.
Before any held-out use, create a different fixed confirmation study comparing
that one frozen winner against uniform at matched training seeds and fidelity.

Never put a password or storage URL in a command, config, artifact, or commit.
Supply the URL, `PGPASSFILE`, and CA path through protected mode-0600 environment
files, and require `sslmode=verify-full`.

## Better next BO design

### Evaluator choice is now resolved

A later matched validation-only bake-off evaluated the six frozen confirmation
checkpoints at three generation seeds with ten fixed Random-GIN evaluators and
ten frozen train-only GraphCL encoders. Random-GIN retained the selected-minus-
uniform sign in all nine training/generation cells and had a mean generation-
seed range of `0.03062`. GraphCL was less dispersed across evaluator instances,
but its generation-seed range was larger (`0.03957`) and its sign stability was
only 8/9. Both methods replayed exactly.

The predeclared decision therefore selects Random-GIN for the next bounded
AIDS KL-weight BO. This strengthens the earlier recommendation: GraphCL is not
being rejected because it is contrastive, but because it was less stable to
the generated collection in this matched AIDS experiment. No held-out graph
was accessed, and this evaluator bake-off makes no new weight-improvement
claim.

The next experiment should improve the objective before merely increasing the
trial count:

1. Measure rank correlation between cheap and full fidelity on several fixed
   anchors. Do not use 100 epochs for selection unless it predicts 250-epoch
   rankings.
2. Score each candidate on at least two matched training seeds during search,
   using their mean as the BO objective. Reserve additional seeds for the final
   confirmation.
3. Parameterize common reconstruction scale and node-to-edge ratio separately,
   or normalize reconstruction components before tuning. This makes the search
   scientifically easier to interpret.
4. Keep uniform as a mandatory anchor and use a 30--50-trial budget only after
   the multi-seed objective and fidelity proxy are qualified.
5. Use successive-halving or promotion only after proving that early scores
   preserve rankings; never promote by looking at test data.
6. Confirm the single frozen winner against uniform at the same final epochs,
   seeds, graph count, evaluator repeats, hardware class, and stopping policy.
7. Report `no_improvement` whenever the paired rule fails. Do not rerank another
   search candidate after seeing confirmation.

PROTEINS can be a later transfer dataset if its frozen split provides enough
validation graphs and both feature decoders are supported, but the same timing,
seed-variance, and fidelity qualifications must precede its BO run.

## Final conclusion

Across both completed real non-QM9 studies, BO has not demonstrated a robust
improvement over uniform GraphVAE reconstruction weights. On AIDS the selected
pair is lower by `0.06449` on mean matched validation Attr-F1PR; on LOBSTER the
selected pair was lower by `0.01153`. Uniform `(1,1)` remains the supported
default. The next BO attempt should optimize a multi-seed, fidelity-qualified
validation objective rather than repeat the same single-seed short-fidelity
search with more trials.
