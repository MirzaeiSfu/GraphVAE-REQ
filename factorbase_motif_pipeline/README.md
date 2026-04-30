# FactorBase Motif Pipeline

This folder contains the scripts used to load graph datasets into MySQL for FactorBase, generate FactorBase config files, launch FactorBase with a selected JAR, and clean up the created databases afterward.

## Main Files

- `run_factorbase_pipeline.py`: wrapper script that runs the dataset import, creates a database-specific config file, and launches FactorBase with the selected JAR.
- `run_qm9_config_compare.py`: helper script that builds QM9 once, clones the prepared database per config, and runs FactorBase for each config.
- `PROTEINS_db.py`: loads the PROTEINS dataset into MySQL.
- `QM9_db.py`: loads the QM9 dataset into MySQL.
- `config.tmp`: template used to generate database-specific config files.
- `drop_factorbase_databases.sh`: helper script to drop a base database and the related FactorBase-created databases.
- `factorbase-1.0-SNAPSHOT.jar`: main FactorBase JAR.
- `factorbase-1.0-patched.jar`: patched FactorBase JAR.
- `factorbase_utils.py`: small shared helper functions used by the wrapper script.

## Generated Artifacts

The pipeline writes generated artifacts into one run folder:

- per-run rule-learning records go in `runs/<db_name>/`
- the generated FactorBase config is `runs/<db_name>/factorbase_config.cfg`
- FactorBase output is moved into `runs/<db_name>/factorbase_output/` after Java finishes
- loose FactorBase files such as `Bif_<db_name>.xml` and `dag_.txt` are moved into `runs/<db_name>/`

Examples:

- `runs/qm9_experiment1/rule_manifest.json`
- `runs/qm9_experiment1/factorbase_config.cfg`
- `runs/qm9_experiment1/command.txt`
- `runs/qm9_experiment1/run.log`
- `runs/qm9_experiment1/Bif_qm9_experiment1.xml`
- `runs/qm9_experiment1/factorbase_output/res/`

The per-database output folder is created by FactorBase itself while Java runs,
then the wrapper moves it under the matching run folder.

FactorBase can then be launched with:

```bash
java -Dconfig=<config-file-path> -jar <jar-file-name>
```

Example:

```bash
java -Dconfig=runs/proteins_experiment/factorbase_config.cfg -jar factorbase-1.0-SNAPSHOT.jar
```

## Current Defaults In `run_factorbase_pipeline.py`

Right now the wrapper script is set to:

- `DEFAULT_DATASET = "LOBSTER"`
- `DEFAULT_CONFIG_TEMPLATE = config.tmp`
- `DEFAULT_JAR = "snapshot"`
- `DEFAULT_EDGE_MODE = "directed"`
- `DEFAULT_SYNTHETIC_EDGE_MODE = "undirected"`

That means:

- if you run the wrapper with no positional arguments, it will use `LOBSTER`
  and compute a database name from the selected options
- it will use undirected edge mode for LOBSTER so the DB matches `main.py`'s symmetric adjacency
- it will run the snapshot JAR because `DEFAULT_JAR` is `"snapshot"`

## Recommended Way To Run

From inside this folder:

```bash
cd /localhome/mirzaei/GraphVAE-REQ/factorbase_motif_pipeline
python run_factorbase_pipeline.py
```

With the current defaults, that will use:

- dataset: `LOBSTER`
- database name: computed automatically, for example `lobster_undir_feat_snap_ab31c9`

If you want to choose the database name yourself, pass it after the dataset:

```bash
python run_factorbase_pipeline.py PROTEINS proteins_experiment
python run_factorbase_pipeline.py QM9 qm9_trial --directed --jar patched
python run_factorbase_pipeline.py QM9 qm9_trial --undirected --jar patched
```

If the dataset database already exists and you only want to rerun FactorBase with a
different config template, you can skip the import step:

```bash
python run_factorbase_pipeline.py QM9 qm9_trial --use-existing-db --jar snapshot --config-template config1.tmp
```

If you want to prepare QM9 once and compare the standard FactorBase jar with
`config.tmp`, `config1.tmp`, and `config2.tmp` on separate cloned databases:

```bash
python run_qm9_config_compare.py --prefix qm9_std_compare
```

## What `run_factorbase_pipeline.py` Does

`run_factorbase_pipeline.py` will:

1. Run the selected dataset import script.
   If you pass `--use-existing-db`, it skips this import step and reuses the current
   `dbname` database instead.
2. Use the provided database name, or compute one from dataset, edge mode,
   feature mode, config template hash, and JAR.
3. Create the MySQL database with that database name.
4. Create a rule-learning record folder at `runs/<db_name>/`.
5. Write `runs/<db_name>/rule_manifest.json`, `factorbase_config.cfg`, `command.txt`, and `run.log`.
7. Clear any old SQL `run_metadata` table before Java starts, because FactorBase scans source tables during setup.
8. Launch FactorBase with `runs/<db_name>/factorbase_config.cfg`.
9. Move the FactorBase-created output folder into `runs/<db_name>/factorbase_output/`.
10. Insert the completed manifest details into the SQL `run_metadata` table inside `<db_name>`.

For example, if you run the pipeline with:

```bash
python run_factorbase_pipeline.py TRIANGULAR_GRID triangular_grid_logmove_smoke --jar snapshot
```

you should expect all of these:

- `runs/triangular_grid_logmove_smoke/rule_manifest.json`
- `runs/triangular_grid_logmove_smoke/factorbase_config.cfg`
- `runs/triangular_grid_logmove_smoke/command.txt`
- `runs/triangular_grid_logmove_smoke/run.log`
- `runs/triangular_grid_logmove_smoke/factorbase_output/`

The manifest records the dataset, edge mode, effective DB edge relation, feature
mode, generated FactorBase config hash, template config hash, selected JAR and
JAR hash, command lines, git commit, git dirty state, and a short reproducibility
hash. It also includes descriptions for source-edge fields such as
`source_edge_rows`, `source_undirected_pairs`, and `source_missing_reverse_rows`.
After FactorBase completes, the SQL database also gets a `run_metadata` table
with the same key fields, so the database can explain how it was created even if
the run folder is moved. `--prepare-only` writes the manifest in the run folder
but does not add `run_metadata`, because extra source tables can confuse
FactorBase setup.

## Dataset Script Notes

- Dataset import scripts accept `--db-name`, `--directed`, and `--undirected`.
- `--directed` preserves the source edge rows; it does not invent directed graph semantics.
- `--undirected` inserts both directions for each source edge pair.
- `--directed` is the default edge mode for `QM9` and `PROTEINS`.
- `--undirected` is the default for synthetic NetworkX datasets: `GRID`, `LOBSTER`, and `TRIANGULAR_GRID`.
- Synthetic datasets reject `--directed` because the DB would store one row per undirected edge while `main.py` uses symmetric adjacency.
- Every dataset import script prints a source-edge bidirectionality analysis before import.
- If every source edge already has its reverse, the scripts warn that `--directed` and `--undirected` should produce the same edge table.
- The current simplified dataset scripts still use the hard-coded MySQL connection values inside those files.
- `QM9_db.py` now uses the repository-level `data/QM9` cache, matching the dataset root used by `main.py`.
- `PROTEINS_db.py` now uses the repository-level `data/dgl` cache when loading through DGL.
- `QM9_db.py` now batches node and edge inserts with `executemany(...)` to reduce SQL round-trips during database population.

## Database Cleanup

`drop_factorbase_databases.sh` supports two modes:

1. Config-file mode
   Give it a generated config file such as `runs/proteins_experiment/factorbase_config.cfg`.
   It reads the MySQL connection settings and `dbname` from that config.

2. Database-name mode
   Give it a plain database name such as `ali`.
   In that case, it uses `config.tmp` only for the MySQL server connection settings and drops:
   - `ali`
   - `ali_setup`
   - `ali_BN`
   - `ali_CT`
   - `ali_global_counts`
   - `ali_CT_cache`

Examples:

```bash
./drop_factorbase_databases.sh --dry-run runs/proteins_experiment/factorbase_config.cfg
./drop_factorbase_databases.sh runs/proteins_experiment/factorbase_config.cfg
./drop_factorbase_databases.sh --dry-run ali
./drop_factorbase_databases.sh ali
```

If you use config-file mode, the script also asks whether you want to delete that config file after the database cleanup.

## Notes

- The wrapper now stores generated configs, logs, manifests, command records, and FactorBase output under `runs/<db_name>/`.
- FactorBase can be pointed at the generated file with `-Dconfig=<config-file-path>`.
- Per-run output folders such as `runs/triangular_grid_logmove_smoke/factorbase_output/` are normal and are created by FactorBase before the wrapper moves them into the run folder.
- Smoke-test folders created during verification can be deleted later if you do not want to keep their outputs.
- Generated run records under `runs/` are ignored by `factorbase_motif_pipeline/.gitignore`.
