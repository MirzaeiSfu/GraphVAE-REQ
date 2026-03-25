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

## Generated Config Files

The wrapper creates config files directly in this folder using the database name, for example:

- `proteins_experiment_config.cfg`
- `qm9_experiment_config.cfg`

FactorBase can then be launched with:

```bash
java -Dconfig=<config-file-name> -jar <jar-file-name>
```

Example:

```bash
java -Dconfig=proteins_experiment_config.cfg -jar factorbase-1.0-SNAPSHOT.jar
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
3. Create a config file named `<db_name>_config.cfg` from `config.tmp`.
4. Ask which JAR to run, unless you set a default or pass `--jar`.
5. Launch FactorBase with `java -Dconfig=<db_name>_config.cfg -jar <selected-jar>`.

## Dataset Script Notes

- `PROTEINS_db.py` and `QM9_db.py` accept `--db-name`, `--directed`, and `--undirected`.
- The current simplified dataset scripts still use the hard-coded MySQL connection values inside those files.
- `QM9_db.py` now uses a script-relative dataset path so it consistently reuses `factorbase_motif_pipeline/data/QM9` instead of depending on the shell working directory.
- `QM9_db.py` now batches node and edge inserts with `executemany(...)` to reduce SQL round-trips during database population.

## Database Cleanup

`drop_factorbase_databases.sh` supports two modes:

1. Config-file mode
   Give it a generated config file such as `proteins_experiment_config.cfg`.
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
./drop_factorbase_databases.sh --dry-run proteins_experiment_config.cfg
./drop_factorbase_databases.sh proteins_experiment_config.cfg
./drop_factorbase_databases.sh --dry-run ali
./drop_factorbase_databases.sh ali
```

If you use config-file mode, the script also asks whether you want to delete that config file after the database cleanup.

## Notes

- The wrapper now uses named config files instead of creating a separate run folder per database name.
- FactorBase can be pointed at the generated file with `-Dconfig=<config-file-name>`.
- Generated config files such as `*_config.cfg` are ignored by `factorbase_motif_pipeline/.gitignore`.
