GraphVAE-REQ cluster pipeline
=============================

This pipeline uses cs-cl-18 as the controller. The workers only need code,
raw data, motif pickle caches, Python/conda/PyTorch/DGL, and GPU access.
Workers do not need MySQL or FactorBase for training.

Important before a real worker run
----------------------------------

The distribution script rsyncs the controller worktree to workers. Workers do
not need GitHub SSH keys. Commit and push the current controller code before
real runs for reproducibility:

  git status --short
  git add main.py CLUSTER_REPO_PATHS.txt CLUSTER_GPU_CONFIGS_SAMPLE.txt CLUSTER_GPU_CONFIGS_MOTIF_SAMPLE.txt CLUSTER_MICRO_PYTHON_PATHS.txt cluster_pipeline configs/cluster_tests scripts/cluster_*.sh CLUSTER_PIPELINE_README.txt
  git commit -m "Update cluster training pipeline"
  git push

The workers receive the current local files from the controller during
distribution, but a git commit makes the exact code version easier to recover.

Input files
-----------

CLUSTER_REPO_PATHS.txt
  Format:
    HOST REPO_PATH

  Example:
    cs-cl-17 /local-scratch/graphvae-req-work/GraphVAE-REQ

CLUSTER_GPU_CONFIGS_SAMPLE.txt
  Format:
    HOST GPU CONFIG_YAML

  Example:
    cs-cl-17 0 configs/reproduce_table2/grid_table2_graphvae_motif.yaml

CLUSTER_MICRO_PYTHON_PATHS.txt
  Format:
    HOST PYTHON_BIN

  Example:
    cs-cl-17 /localhome/mirzaei/miniconda3/envs/micro/bin/python

  This lets the runner use different absolute micro paths on different hosts.

Lines may contain comments after #. Blank lines are ignored.

Recommended order
-----------------

1. Prepare fresh motif caches on the controller.
2. Distribute code, raw data, and motif caches to workers.
3. Launch the scheduled tmux training jobs.
4. Collect run outputs back to the controller.

Use --dry-run first for every step.


1. Prepare motif caches
-----------------------

Script:
  scripts/cluster_prepare_motif_caches.sh

Purpose:
  Rebuild local motif pickle caches on the controller. This is the only step
  that needs FactorBase/MySQL.

Default command:
  scripts/cluster_prepare_motif_caches.sh

Dry run:
  scripts/cluster_prepare_motif_caches.sh --dry-run

Typical explicit command:
  scripts/cluster_prepare_motif_caches.sh \
    --schedule CLUSTER_GPU_CONFIGS_SAMPLE.txt \
    --motif-cache-dir cache_motifs

Useful options:
  --schedule FILE
    Schedule file with rows: HOST GPU CONFIG_YAML.

  --python-bin BIN
    Python executable for running main.py.

  --motif-cache-dir DIR
    Directory where fresh motif pickles are written. Default: cache_motifs.

  --archive-root DIR
    Directory where an old motif cache folder is moved before rebuild.
    Default: cache_motifs_archive.

  --manifest FILE
    Manifest output path. Default:
      <motif-cache-dir>/MOTIF_CACHE_MANIFEST.tsv

What it does:
  - Archives the old cache_motifs directory if it exists.
  - Creates a fresh cache_motifs directory.
  - Reads each unique YAML from the schedule.
  - Runs only configs with motif_loss: true.
  - Runs main.py with --prepare_motif_cache_only true.
  - Writes a TSV manifest with motif pickle path, sha256, size, and mtime.

Notes:
  Non-motif configs are skipped because they do not need motif pickles.


2. Distribute code and inputs
-----------------------------

Script:
  scripts/cluster_distribute_code.sh

Purpose:
  Rsync the controller code to each worker, then optionally sync raw data and
  motif caches.

Dry run, code only:
  scripts/cluster_distribute_code.sh --dry-run

Dry run, code plus inputs:
  scripts/cluster_distribute_code.sh --dry-run --sync-inputs

Real command after motif caches exist:
  scripts/cluster_distribute_code.sh \
    --repo-paths CLUSTER_REPO_PATHS.txt \
    --sync-inputs

Useful options:
  --repo-paths FILE
    Repo path file with rows: HOST REPO_PATH.

  --code-source DIR
    Controller repo directory to sync. Default: current directory.

  --sync-inputs
    Sync the default input folders:
      data_raw
      cache_motifs

  --sync-path PATH
    Replace the default sync list with explicit paths. Repeatable.

  --ssh-connect-timeout SEC
    SSH connection timeout. Default: 10.

What it does:
  - Refuses real distribution if the controller git worktree is dirty.
  - For each host, creates the worker repo directory if missing.
  - Rsyncs controller code into the worker repo.
  - Excludes .git, data_raw, cache_motifs, cache/archive folders, runs, and
    collected outputs from the code sync.
  - With --sync-inputs, rsyncs data_raw and cache_motifs.
  - Before syncing cache_motifs, deletes the worker's old cache_motifs folder.
  - Uses checksum rsync for cache_motifs.
  - Continues with other hosts if one host fails.

Notes:
  Dataset caches are not synced. Real worker training disables dataset caches
  and reads/processes the raw data directly.

  Dry-run prints dirty-worktree warnings, but real distribute exits with an
  error until the controller changes are committed, stashed, or removed.

  SSH uses StrictHostKeyChecking=accept-new, so a first-time short hostname
  alias such as cs-cl-16 can be added automatically while changed known host
  keys are still rejected.


3. Launch scheduled training
----------------------------

Script:
  scripts/cluster_run_schedule.sh

Purpose:
  Start one tmux training session per schedule row.

Dry run:
  scripts/cluster_run_schedule.sh --dry-run --date-prefix YYYYMMDD

Real command:
  scripts/cluster_run_schedule.sh \
    --repo-paths CLUSTER_REPO_PATHS.txt \
    --schedule CLUSTER_GPU_CONFIGS_SAMPLE.txt \
    --date-prefix YYYYMMDD

Useful options:
  --repo-paths FILE
    Repo path file with rows: HOST REPO_PATH.

  --schedule FILE
    Schedule file with rows: HOST GPU CONFIG_YAML.

  --date-prefix YYYYMMDD
    Date prefix for run folder names. Default is today's date.

  --run-root PATH
    Output root inside each worker repo. Default: runs/distributed.
    The ./cluster_pipeline wrapper uses:
      runs/<YYYYMMDD>/cluster_smoke_grid_motif

  --python-bin BIN
    Fallback Python executable on workers.

  --python-paths FILE
    Optional host-specific Python file with rows: HOST PYTHON_BIN.
    Matching rows override --python-bin.

  --env-activate CMD
    Optional environment activation command on workers.
    Example:
      --env-activate "source ~/miniconda3/etc/profile.d/conda.sh && conda activate graphvae"

  --ssh-connect-timeout SEC
    SSH connection timeout. Default: 10.

  -- extra main.py args
    Arguments after -- are appended to the main.py command.

What it does:
  - Reads the repo path file.
  - Reads the schedule file.
  - Builds one run folder per row:
      <run-root>/<config-name>__<host>_gpu<gpu>
  - Starts a detached tmux session on the worker.
  - Writes stdout/stderr to:
      <run-folder>/stdout.log
  - Writes run metadata to:
      <run-folder>/RUN_INFO.txt
  - Passes --disable_dataset_cache true by default.
  - If SSH fails for a host, later rows for that host are skipped.

Notes:
  Date-only names are simple, but the same schedule should not be launched
  twice with the same date prefix and run root. For same-day repeats, use a
  different --run-root.

  To enable dataset cache for debugging only:
    scripts/cluster_run_schedule.sh --dry-run -- \
      --disable_dataset_cache false \
      --dataset_cache_dir cache_datasets


4. Collect results
------------------

Script:
  scripts/cluster_collect_results.sh

Purpose:
  Copy worker run outputs back to the controller.

Dry run:
  scripts/cluster_collect_results.sh --dry-run --date-prefix YYYYMMDD

Real command:
  scripts/cluster_collect_results.sh \
    --repo-paths CLUSTER_REPO_PATHS.txt \
    --date-prefix YYYYMMDD

Useful options:
  --repo-paths FILE
    Repo path file with rows: HOST REPO_PATH.

  --remote-run-root PATH
    Remote run root inside each worker repo. Default: runs/distributed.
    The ./cluster_pipeline wrapper passes:
      runs/<YYYYMMDD>/cluster_smoke_grid_motif

  --collect-root PATH
    Local collection root. Default: collected_runs.

  --date-prefix YYYYMMDD
    Local collection batch folder. Default is today's date.

  --ssh-connect-timeout SEC
    SSH connection timeout. Default: 10.

What it does:
  - Rsyncs each worker's remote run root to:
      collected_runs/<YYYYMMDD>/<experiment-name>/
  - Continues with other hosts if one host fails.

Notes:
  The date prefix controls the local collection folder. With ./cluster_pipeline,
  the worker-side date is also part of the remote run root, so collect targets
  only the current day's experiment folder:
      runs/<YYYYMMDD>/cluster_smoke_grid_motif

  For the smoke wrapper, the collected layout is:
      collected_runs/<YYYYMMDD>/cluster_smoke_grid_motif/


Smoke-test helper
-----------------

The repository also includes a small wrapper:

  ./cluster_pipeline

Mixed motif GRID smoke test:

  ./cluster_pipeline dry-run
  ./cluster_pipeline prepare
  ./cluster_pipeline distribute
  ./cluster_pipeline run
  ./cluster_pipeline collect

This uses:

  CLUSTER_GPU_CONFIGS_MOTIF_SAMPLE.txt
  CLUSTER_MICRO_PYTHON_PATHS.txt
  configs/cluster_tests/grid_2epoch_graphvae_motif.yaml
  configs/cluster_tests/grid_2epoch_graphvae_baseline.yaml

The active wrapper assumes at least one scheduled job has motif_loss: true.
Run prepare before distribute, because distribute syncs both data_raw and
cache_motifs and requires real cache_motifs/*.pkl files.

The wrapper reads CLUSTER_MICRO_PYTHON_PATHS.txt by default:

  MICRO_PYTHON_PATHS=CLUSTER_MICRO_PYTHON_PATHS.txt ./cluster_pipeline dry-run

It uses the controller host's row for motif cache preparation, and passes the
same file to the worker runner so each host uses its own recorded micro path.
PYTHON_BIN is still accepted as a controller override and worker fallback:

  PYTHON_BIN=/path/to/python ./cluster_pipeline dry-run

The wrapper writes worker outputs under:

  runs/<YYYYMMDD>/cluster_smoke_grid_motif/

Each job folder uses the YAML name plus host/GPU:

  grid_2epoch_graphvae_motif__cs-cl-17_gpu0/

Each job folder contains RUN_INFO.txt with fields such as date_prefix,
run_root, config_name, config_path, host, gpu, device, and python_bin.

and collects them back under:

  collected_runs/<YYYYMMDD>/cluster_smoke_grid_motif/

Current paths include /localhome/mirzaei/miniconda3/envs/micro/bin/python on
most hosts, and /local-scratch2/mirzaei/miniconda3/envs/micro/bin/python on
cs-cl-18 and cs-cl-19.

cs-cl-18 is kept as the controller and is intentionally removed from worker
schedules for now because CUDA/GPU visibility was not working during the
2026-06-29 check.

The current motif sample runs motif_loss: true only on:

  cs-cl-17 GPU 0
  cs-cl-17 GPU 1

All other scheduled GPUs run the non-motif baseline test.

All-non-motif variant:

  If every scheduled YAML has motif_loss: false, use CLUSTER_GPU_CONFIGS_SAMPLE.txt
  and sync data_raw only. The code for that case is kept as a commented block
  in ./cluster_pipeline.


Quick full dry-run checklist
----------------------------

Run these from the controller repo:

  bash -n scripts/cluster_prepare_motif_caches.sh scripts/cluster_distribute_code.sh scripts/cluster_run_schedule.sh scripts/cluster_collect_results.sh
  python -m py_compile main.py
  scripts/cluster_prepare_motif_caches.sh --dry-run
  scripts/cluster_distribute_code.sh --dry-run
  scripts/cluster_run_schedule.sh --dry-run --date-prefix YYYYMMDD
  scripts/cluster_collect_results.sh --dry-run --date-prefix YYYYMMDD

After real motif cache preparation, this should also pass:

  scripts/cluster_distribute_code.sh --dry-run --sync-inputs
