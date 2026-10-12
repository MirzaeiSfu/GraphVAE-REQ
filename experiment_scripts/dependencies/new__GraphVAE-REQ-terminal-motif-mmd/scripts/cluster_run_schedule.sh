#!/usr/bin/env bash
set -u

REPO_PATHS_FILE="CLUSTER_REPO_PATHS.txt"
SCHEDULE_FILE="CLUSTER_GPU_CONFIGS_SAMPLE.txt"
DATE_PREFIX="$(date +%Y%m%d)"
RUN_ROOT="runs/distributed"
PYTHON_BIN="python"
PYTHON_PATHS_FILE=""
ENV_ACTIVATE=""
SSH_CONNECT_TIMEOUT=10
DRY_RUN=false
TRAIN_ARGS=()

usage() {
  cat <<'EOF'
Start one tmux training job per schedule row.

Cluster jobs disable processed dataset caches by default. They read/process
the raw dataset files for each run, which avoids stale shared caches and keeps
dataset cache pickles out of worker outputs.

Usage:
  scripts/cluster_run_schedule.sh [options] [-- extra main.py args]

Options:
  --repo-paths FILE          Input file with rows: HOST REPO_PATH
  --schedule FILE            Input file with rows: HOST GPU CONFIG_YAML
  --date-prefix YYYYMMDD     Prefix for run names; default is today
  --run-root PATH            Output root inside each repo
  --python-bin BIN           Fallback Python executable on workers
  --python-paths FILE        Optional file with rows: HOST PYTHON_BIN
  --env-activate CMD         Worker environment activation command
  --ssh-connect-timeout SEC  SSH connection timeout
  --dry-run                  Print commands without running them
  --help                     Show this help
EOF
}

quote_words() {
  local quoted=()
  local word
  local word_q
  for word in "$@"; do
    printf -v word_q '%q' "$word"
    quoted+=("$word_q")
  done
  printf '%s' "${quoted[*]}"
}

quote_one() {
  printf '%q' "$1"
}

sanitize() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9_.-]/_/g; s/^[._-]*//; s/[._-]*$//'
}

line_is_blank() {
  [[ -z "${1//[[:space:]]/}" ]]
}

run_cmd() {
  echo "+ $(quote_words "$@")"
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi
  "$@"
}

while (($#)); do
  case "$1" in
    --repo-paths)
      REPO_PATHS_FILE="$2"
      shift 2
      ;;
    --schedule)
      SCHEDULE_FILE="$2"
      shift 2
      ;;
    --date-prefix)
      DATE_PREFIX="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --python-paths)
      PYTHON_PATHS_FILE="$2"
      shift 2
      ;;
    --env-activate)
      ENV_ACTIVATE="$2"
      shift 2
      ;;
    --ssh-connect-timeout)
      SSH_CONNECT_TIMEOUT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      TRAIN_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$DATE_PREFIX" =~ ^[0-9]{8}$ ]]; then
  echo "--date-prefix must look like YYYYMMDD" >&2
  exit 2
fi

if [[ ! -f "$REPO_PATHS_FILE" ]]; then
  echo "Missing repo paths file: $REPO_PATHS_FILE" >&2
  exit 2
fi

if [[ ! -f "$SCHEDULE_FILE" ]]; then
  echo "Missing schedule file: $SCHEDULE_FILE" >&2
  exit 2
fi

declare -A REPO_PATH_BY_HOST
declare -A PYTHON_BY_HOST
declare -A HOST_IS_DOWN

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line%%#*}"
  line_is_blank "$line" && continue

  read -r host repo_path extra <<<"$line"
  if [[ -n "${extra:-}" || -z "${host:-}" || -z "${repo_path:-}" ]]; then
    echo "[repo-paths] bad row: $raw_line" >&2
    continue
  fi
  REPO_PATH_BY_HOST["$host"]="$repo_path"
done < "$REPO_PATHS_FILE"

if [[ -n "$PYTHON_PATHS_FILE" ]]; then
  if [[ ! -f "$PYTHON_PATHS_FILE" ]]; then
    echo "Missing python paths file: $PYTHON_PATHS_FILE" >&2
    exit 2
  fi

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%%#*}"
    line_is_blank "$line" && continue

    read -r host python_path extra <<<"$line"
    if [[ -n "${extra:-}" || -z "${host:-}" || -z "${python_path:-}" ]]; then
      echo "[python-paths] bad row: $raw_line" >&2
      continue
    fi
    PYTHON_BY_HOST["$host"]="$python_path"
  done < "$PYTHON_PATHS_FILE"
fi

launched=0
failures=0
SSH_OPTS=(-o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" -o StrictHostKeyChecking=accept-new)

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line%%#*}"
  line_is_blank "$line" && continue

  read -r host gpu config_path extra <<<"$line"
  if [[ -n "${extra:-}" || -z "${host:-}" || -z "${gpu:-}" || -z "${config_path:-}" ]]; then
    echo "[run] bad schedule row: $raw_line" >&2
    failures=$((failures + 1))
    continue
  fi

  if [[ -n "${HOST_IS_DOWN[$host]:-}" ]]; then
    echo "[run] skip $host GPU $gpu because SSH failed earlier"
    failures=$((failures + 1))
    continue
  fi

  repo_path="${REPO_PATH_BY_HOST[$host]:-}"
  if [[ -z "$repo_path" ]]; then
    echo "[run] no repo path for host $host" >&2
    failures=$((failures + 1))
    continue
  fi

  job_python_bin="$PYTHON_BIN"
  if [[ -n "${PYTHON_BY_HOST[$host]:-}" ]]; then
    job_python_bin="${PYTHON_BY_HOST[$host]}"
  fi

  if [[ "$job_python_bin" == "NOT_FOUND" ]]; then
    echo "[run] no micro Python path for host $host" >&2
    failures=$((failures + 1))
    continue
  fi

  if [[ ! -f "$config_path" ]]; then
    echo "[run] local config missing: $config_path" >&2
    failures=$((failures + 1))
    continue
  fi

  experiment_name="$(sanitize "$(basename "$config_path" .yaml)")"
  host_token="$(sanitize "$host")"
  gpu_token="$(sanitize "$gpu")"
  device="cuda:$gpu"
  if [[ "$gpu" == cpu || "$gpu" == cuda:* ]]; then
    device="$gpu"
  fi

  job_dir="${experiment_name}__${host_token}_gpu${gpu_token}"
  run_dir="${RUN_ROOT%/}/${job_dir}"
  run_label="${DATE_PREFIX}_${job_dir}"
  session_name="$(sanitize "$run_label")"

  main_cmd=(
    env
    MPLBACKEND=Agg
    PYTHONUNBUFFERED=1
    "$job_python_bin"
    -u
    main.py
    --config "$config_path"
    --device "$device"
    --graph_save_path "$run_dir"
    --run_label "$run_label"
    --disable_dataset_cache true
  )
  if ((${#TRAIN_ARGS[@]})); then
    main_cmd+=("${TRAIN_ARGS[@]}")
  fi

  repo_q="$(quote_one "$repo_path")"
  python_q="$(quote_one "$job_python_bin")"
  run_dir_q="$(quote_one "$run_dir")"
  run_info_q="$(quote_one "$run_dir/RUN_INFO.txt")"
  stdout_q="$(quote_one "$run_dir/stdout.log")"
  run_info_cmd=(
    printf '%s\n'
    "date_prefix=$DATE_PREFIX"
    "run_root=$RUN_ROOT"
    "job_dir=$job_dir"
    "run_dir=$run_dir"
    "run_label=$run_label"
    "config_name=$experiment_name"
    "config_path=$config_path"
    "host=$host"
    "gpu=$gpu"
    "device=$device"
    "python_bin=$job_python_bin"
  )
  session_cmd=$(cat <<EOF
set -euo pipefail
${ENV_ACTIVATE}
cd $repo_q
mkdir -p $run_dir_q
$(quote_words "${run_info_cmd[@]}") > $run_info_q
$(quote_words "${main_cmd[@]}") 2>&1 | tee $stdout_q
EOF
)

  remote_script=$(cat <<EOF
set -euo pipefail
if ! command -v tmux >/dev/null 2>&1; then
  echo 'tmux is not installed on this host.' >&2
  exit 20
fi
if [[ $python_q == */* ]]; then
  if [[ ! -x $python_q ]]; then
    echo 'Python executable is missing or not executable: $job_python_bin' >&2
    exit 22
  fi
else
  if ! command -v $python_q >/dev/null 2>&1; then
    echo 'Python command was not found: $job_python_bin' >&2
    exit 22
  fi
fi
cd $repo_q
if tmux has-session -t $(quote_one "$session_name") 2>/dev/null; then
  echo 'tmux session already exists: $session_name' >&2
  exit 21
fi
mkdir -p $run_dir_q
tmux new-session -d -s $(quote_one "$session_name") bash -lc $(quote_one "$session_cmd")
EOF
)

  echo
  echo "[run] $host $device $config_path"
  echo "[run] python: $job_python_bin"
  echo "[run] output: $run_dir"

  ssh_cmd=(ssh -n "${SSH_OPTS[@]}" "$host" "bash -lc $(quote_one "$remote_script")")
  if run_cmd "${ssh_cmd[@]}"; then
    launched=$((launched + 1))
    remote_log_path="${repo_path%/}/$run_dir/stdout.log"
    echo "[logs] ssh $host -t $(quote_one "tail -f $remote_log_path")"
  else
    status=$?
    failures=$((failures + 1))
    if [[ "$status" -eq 255 ]]; then
      HOST_IS_DOWN["$host"]=1
      echo "[run] SSH failed on $host; remaining jobs on this host will be skipped" >&2
    else
      echo "[run] failed only this job; continuing" >&2
    fi
  fi
done < "$SCHEDULE_FILE"

echo
echo "[summary] launched: $launched"
echo "[summary] failures/skipped: $failures"
exit "$failures"
