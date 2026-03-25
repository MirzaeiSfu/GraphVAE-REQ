# FactorBase Motif Pipeline

This folder contains the scripts used to load graph datasets into MySQL for FactorBase, generate FactorBase config files, launch FactorBase with a selected JAR, and clean up the created databases afterward.

## Main Files

- `run_factorbase_pipeline.py`: wrapper script that runs the dataset import, creates a database-specific config file, and launches FactorBase with the selected JAR.
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

- `DEFAULT_DATASET = "QM9"`
- `DEFAULT_DB_NAME = "qm9_experiment1"`
- `DEFAULT_CONFIG_TEMPLATE = config.tmp`
- `DEFAULT_JAR = None`
- `DEFAULT_EDGE_MODE = None`

That means:

- if you run the wrapper with no positional arguments, it will use `QM9` and `qm9_experiment1`
- it will still ask for edge mode because `DEFAULT_EDGE_MODE` is `None`
- it will still ask which JAR to run because `DEFAULT_JAR` is `None`

## Recommended Way To Run

From inside this folder:

```bash
cd /localhome/mirzaei/GraphVAE-REQ/factorbase_motif_pipeline
python run_factorbase_pipeline.py
```

With the current defaults, that will use:

- dataset: `QM9`
- database name: `qm9_experiment1`

If you want to override the defaults:

```bash
python run_factorbase_pipeline.py PROTEINS proteins_experiment
python run_factorbase_pipeline.py QM9 qm9_trial --undirected --jar patched
```

## What `run_factorbase_pipeline.py` Does

`run_factorbase_pipeline.py` will:

1. Run either `PROTEINS_db.py` or `QM9_db.py`.
2. Create the MySQL database with the chosen database name.
3. Create a config file named `config/<db_name>_config.cfg` from `config.tmp`.
4. Ask which JAR to run, unless you set a default or pass `--jar`.
5. Write a pipeline log in `log/<db_name>_run.log`.
6. Launch FactorBase with `java -Dconfig=config/<db_name>_config.cfg -jar <selected-jar>`.
7. Let FactorBase create its own output folder named `<db_name>/`.

For example, if you run the pipeline with:

```bash
python run_factorbase_pipeline.py TRIANGULAR_GRID triangular_grid_logmove_smoke --undirected --jar snapshot
```

you should expect all of these:

- `config/triangular_grid_logmove_smoke_config.cfg`
- `log/triangular_grid_logmove_smoke_run.log`
- `triangular_grid_logmove_smoke/`

## Dataset Script Notes

- `PROTEINS_db.py` and `QM9_db.py` accept `--db-name`, `--directed`, and `--undirected`.
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
