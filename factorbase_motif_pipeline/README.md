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

The pipeline now writes generated artifacts into a few different places:

- config files go in `config/`
- pipeline wrapper logs go in `log/`
- FactorBase still creates a run-output folder named after the database, for example `qm9_experiment1/` or `triangular_grid_logmove_smoke/`

Examples:

- `config/proteins_experiment_config.cfg`
- `config/qm9_experiment_config.cfg`
- `log/qm9_experiment1_run.log`
- `triangular_grid_logmove_smoke/res/`

The per-database output folder is created by FactorBase itself. It is where run artifacts such as `res/*.xml` are written.

FactorBase can then be launched with:

```bash
java -Dconfig=<config-file-path> -jar <jar-file-name>
```

Example:

```bash
java -Dconfig=config/proteins_experiment_config.cfg -jar factorbase-1.0-SNAPSHOT.jar
```

## Current Defaults In `run_factorbase_pipeline.py`

Right now the wrapper script is set to:

- `DEFAULT_DATASET = "LOBSTER"`
- `DEFAULT_DB_NAME = "lobster_experiment"`
- `DEFAULT_CONFIG_TEMPLATE = config.tmp`
- `DEFAULT_JAR = "snapshot"`
- `DEFAULT_EDGE_MODE = "directed"`
- `DEFAULT_SYNTHETIC_EDGE_MODE = "undirected"`

That means:

- if you run the wrapper with no positional arguments, it will use `LOBSTER` and `lobster_experiment`
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
- database name: `lobster_experiment`

If you want to override the defaults:

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
2. Create the MySQL database with the chosen database name.
3. Create a config file named `config/<db_name>_config.cfg` from `config.tmp`.
4. Ask which JAR to run, unless you set a default or pass `--jar`.
5. Write a pipeline log in `log/<db_name>_run.log`.
6. Launch FactorBase with `java -Dconfig=config/<db_name>_config.cfg -jar <selected-jar>`.
7. Let FactorBase create its own output folder named `<db_name>/`.

For example, if you run the pipeline with:

```bash
python run_factorbase_pipeline.py TRIANGULAR_GRID triangular_grid_logmove_smoke --jar snapshot
```

you should expect all of these:

- `config/triangular_grid_logmove_smoke_config.cfg`
- `log/triangular_grid_logmove_smoke_run.log`
- `triangular_grid_logmove_smoke/`

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
- `QM9_db.py` now uses a script-relative dataset path so it consistently reuses `factorbase_motif_pipeline/data/QM9` instead of depending on the shell working directory.
- `QM9_db.py` now batches node and edge inserts with `executemany(...)` to reduce SQL round-trips during database population.

## Database Cleanup

`drop_factorbase_databases.sh` supports two modes:

1. Config-file mode
   Give it a generated config file such as `config/proteins_experiment_config.cfg`.
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
./drop_factorbase_databases.sh --dry-run config/proteins_experiment_config.cfg
./drop_factorbase_databases.sh config/proteins_experiment_config.cfg
./drop_factorbase_databases.sh --dry-run ali
./drop_factorbase_databases.sh ali
```

If you use config-file mode, the script also asks whether you want to delete that config file after the database cleanup.

## Notes

- The wrapper now stores generated config files in `config/` and run logs in `log/`.
- FactorBase can be pointed at the generated file with `-Dconfig=<config-file-path>`.
- Per-run output folders such as `triangular_grid_logmove_smoke/` are normal and are created by FactorBase, not by the config/log reorganization itself.
- Smoke-test folders created during verification can be deleted later if you do not want to keep their outputs.
- Generated config files such as `config/*_config.cfg` and logs such as `log/*.log` are ignored by `factorbase_motif_pipeline/.gitignore`.
