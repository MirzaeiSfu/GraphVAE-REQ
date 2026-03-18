# FactorBase Motif Pipeline

This folder contains the dataset-to-MySQL scripts, the FactorBase launcher script, the config template, the JAR files, and some generated run directories.

## Current Main Files

- `run_factorbase_pipeline.py`: main wrapper script that runs the dataset import, creates a run folder, writes `config.cfg`, copies the JARs, and launches FactorBase.
- `PROTEINS_db.py`: loads the PROTEINS dataset into MySQL.
- `QM9_db.py`: loads the QM9 dataset into MySQL.
- `config.tmp`: template used to generate `config.cfg` inside each run folder.
- `drop_factorbase_databases.sh`: helper script to drop the base database and the FactorBase-created databases.
- `factorbase-1.0-SNAPSHOT.jar`: main FactorBase JAR.
- `factorbase-1.0-patched.jar`: patched FactorBase JAR.
- `factorbase_utils.py`: small shared helper functions used by the wrapper script.

## Current Run Folders In This Directory

At the moment this folder also contains these run directories:

- `factorbase_PROTEINS_run`
- `factorbase_QM9_run`
- `factorbase_proteins_experiment_run`

These are run-specific folders that contain their own `config.cfg` and any generated FactorBase outputs.

## Current Defaults In `run_factorbase_pipeline.py`

Right now the wrapper script is set to:

- `DEFAULT_DATASET = "QM9"`
- `DEFAULT_DB_NAME = "qm9_experiment"`
- `DEFAULT_CONFIG_TEMPLATE = config.tmp`
- `DEFAULT_RUN_DIR = None`
- `DEFAULT_JAR = None`
- `DEFAULT_EDGE_MODE = None`

That means:

- if you run the wrapper with no positional arguments, it will use `QM9` and `qm9_experiment`
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
- database name: `qm9_experiment`

If you want to override the defaults:

```bash
python run_factorbase_pipeline.py PROTEINS proteins_experiment
python run_factorbase_pipeline.py QM9 qm9_trial --undirected --jar patched
```

## What The Wrapper Does

`run_factorbase_pipeline.py` will:

1. Run either `PROTEINS_db.py` or `QM9_db.py`.
2. Create the MySQL database with the chosen database name.
3. Create a run folder named `factorbase_<db_name>_run`.
4. Create `config.cfg` from `config.tmp`.
5. Copy both JAR files into that run folder.
6. Ask which JAR to run, unless you set a default or pass `--jar`.

## Notes

- `PROTEINS_db.py` and `QM9_db.py` accept `--db-name`, `--directed`, and `--undirected`.
- The current simplified dataset scripts still use the hard-coded MySQL connection values inside those files.
- FactorBase expects the config file inside each run folder to be named exactly `config.cfg`.
