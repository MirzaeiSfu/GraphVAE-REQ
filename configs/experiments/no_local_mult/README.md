# GraphVAE+RG configs with the `no_local_mult` rule-pruning score

One file per paper run (`<DATASET>/graphvae_rg/seed_N.yaml`), identical to
`configs/paper_runs/<DATASET>/graphvae_rg/seed_N.yaml` except for one added setting:

```yaml
rule_prune_score: no_local_mult   # score 2*(ln CP - ln prior) instead of 2*local_mult*(ln CP - ln prior) - ln local_mult
```

```bash
python main.py --config configs/experiments/no_local_mult/GRID/graphvae_rg/seed_0.yaml
```

- The paper configs in `configs/paper_runs/` are unchanged (default `rule_prune_score: original`).
- Pruning only applies when `rule_prune` is true. QM9's paper config has `rule_prune: 'false'`, so for QM9 this
  setting changes nothing unless pruning is also turned on (`--rule_prune True`).
- Use `--seed N` (and, for PTC, PROTEINS and QM9, `--third_party_eval_seed N`) for a new seed.
- PTC, PROTEINS and QM9 configs keep the original machines' paths: pass `--graph_save_path`, `--dataset_cache_dir`,
  `--motif_cache_dir` (and `--data_dir` for QM9) on the command line.
