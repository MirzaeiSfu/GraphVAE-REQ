#!/usr/bin/env bash
set -u

REPO_PATHS_FILE="CLUSTER_REPO_PATHS.txt"
PYTHON_PATHS_FILE="CLUSTER_MICRO_PYTHON_PATHS.txt"
SCHEDULE_FILE="CLUSTER_GRAPHCL_GIN_20260725.txt"
INPUT_ROOT="graph_evaluation_inputs/20260725"
RUN_ROOT="runs/graphcl_gin_20260725"
UPSTREAM_NAME="Self-Supervised-Models-for-GGM-Evaluation"
DEPS_DIR=".graphcl_deps"
EPOCHS=100
SEEDS=(0 1 2)
SSH_CONNECT_TIMEOUT=10
DRY_RUN=false

usage() {
  cat <<'EOF'
Launch one GraphCL-GIN training job per dataset in a remote tmux session.

Schedule rows:
  HOST GPU DATASET FEATURE_MODE

Each job consumes:
  <repo>/<input-root>/<dataset>/real_train_graphs.pt

The pinned contrastive upstream must be a sibling of the worker repository,
and the isolated PyGCL dependency bundle must exist at <repo>/<deps-dir>.

Usage:
  scripts/cluster_run_graphcl_schedule.sh [options]

Options:
  --repo-paths FILE          HOST REPO_PATH mapping
  --python-paths FILE        HOST PYTHON_BIN mapping
  --schedule FILE            GraphCL schedule
  --input-root PATH          Input artifact root inside each worker repo
  --run-root PATH            Output root inside each worker repo
  --upstream-name NAME       Sibling checkout directory name
  --deps-dir PATH            Isolated dependency path inside worker repo
  --epochs N                 Contrastive training epochs; default: 100
  --seeds LIST               Comma-separated independent seeds; default: 0,1,2
  --ssh-connect-timeout SEC  SSH connection timeout
  --dry-run                  Print commands without launching
  --help                     Show this help
EOF
}

quote_one() {
  printf '%q' "$1"
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

line_is_blank() {
  [[ -z "${1//[[:space:]]/}" ]]
}

sanitize() {
  printf '%s' "$1" |
    sed 's/[^A-Za-z0-9_.-]/_/g; s/^[._-]*//; s/[._-]*$//'
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
    --python-paths)
      PYTHON_PATHS_FILE="$2"
      shift 2
      ;;
    --schedule)
      SCHEDULE_FILE="$2"
      shift 2
      ;;
    --input-root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --upstream-name)
      UPSTREAM_NAME="$2"
      shift 2
      ;;
    --deps-dir)
      DEPS_DIR="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --seeds)
      IFS=',' read -r -a SEEDS <<<"$2"
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
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for required in "$REPO_PATHS_FILE" "$PYTHON_PATHS_FILE" "$SCHEDULE_FILE"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing input file: $required" >&2
    exit 2
  fi
done
if [[ ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--epochs must be a positive integer." >&2
  exit 2
fi
if ((${#SEEDS[@]} == 0)); then
  echo "--seeds must contain at least one integer." >&2
  exit 2
fi
for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Bad seed: $seed" >&2
    exit 2
  fi
done

declare -A REPO_BY_HOST
declare -A PYTHON_BY_HOST
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line%%#*}"
  line_is_blank "$line" && continue
  read -r host repo extra <<<"$line"
  if [[ -z "${host:-}" || -z "${repo:-}" || -n "${extra:-}" ]]; then
    echo "Bad repo-path row: $raw_line" >&2
    exit 2
  fi
  REPO_BY_HOST["$host"]="$repo"
done < "$REPO_PATHS_FILE"
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line%%#*}"
  line_is_blank "$line" && continue
  read -r host python_bin extra <<<"$line"
  if [[ -z "${host:-}" || -z "${python_bin:-}" || -n "${extra:-}" ]]; then
    echo "Bad Python-path row: $raw_line" >&2
    exit 2
  fi
  PYTHON_BY_HOST["$host"]="$python_bin"
done < "$PYTHON_PATHS_FILE"

SSH_OPTS=(
  -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT"
  -o StrictHostKeyChecking=accept-new
)
launched=0
failures=0

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line%%#*}"
  line_is_blank "$line" && continue
  read -r host gpu dataset feature_mode extra <<<"$line"
  if [[ -z "${host:-}" || -z "${gpu:-}" || -z "${dataset:-}" ||
        -z "${feature_mode:-}" || -n "${extra:-}" ]]; then
    echo "Bad schedule row: $raw_line" >&2
    failures=$((failures + 1))
    continue
  fi
  case "$feature_mode" in
    topology_control|decoded_node|decoded_edge|decoded_node_edge) ;;
    *)
      echo "Bad feature mode in row: $raw_line" >&2
      failures=$((failures + 1))
      continue
      ;;
  esac

  repo="${REPO_BY_HOST[$host]:-}"
  python_bin="${PYTHON_BY_HOST[$host]:-}"
  if [[ -z "$repo" || -z "$python_bin" || "$python_bin" == "NOT_FOUND" ]]; then
    echo "Missing repo or Python mapping for $host" >&2
    failures=$((failures + 1))
    continue
  fi

  dataset_token="$(sanitize "$dataset")"
  host_token="$(sanitize "$host")"
  session_name="graphcl_${dataset_token}_20260725"
  run_dir="${RUN_ROOT%/}/${dataset_token}__${host_token}_gpu${gpu}"
  graph_path="${INPUT_ROOT%/}/${dataset}/real_train_graphs.pt"
  parent="$(dirname "$repo")"
  upstream="$parent/$UPSTREAM_NAME"
  deps="$repo/$DEPS_DIR"
  output="$repo/$run_dir"
  graph_abs="$repo/$graph_path"
  stdout="$output/stdout.log"
  exit_status="$output/exit_status.txt"
  run_info="$output/RUN_INFO.txt"
  seeds_text="${SEEDS[*]}"

  train_cmd=(
    env
    "PYTHONPATH=$deps:$repo/graph_evaluation/src"
    PYTHONUNBUFFERED=1
    MPLBACKEND=Agg
    "$python_bin"
    -u
    -m
    ggm_eval.cli
    train
    --graphs "$graph_abs"
    --encoder graphcl
    --feature-mode "$feature_mode"
    --seeds "${SEEDS[@]}"
    --epochs "$EPOCHS"
    --num-layers 3
    --hidden-dim 32
    --init orthogonal
    --limit-lipschitz
    --device "cuda:$gpu"
    --python "$python_bin"
    --upstream-repo "$upstream"
    --output-dir "$output"
  )

  output_q="$(quote_one "$output")"
  stdout_q="$(quote_one "$stdout")"
  exit_status_q="$(quote_one "$exit_status")"
  run_info_q="$(quote_one "$run_info")"
  session_cmd=$(cat <<EOF
set -uo pipefail
mkdir -p $output_q
printf '%s\n' \
  'dataset=$dataset' \
  'feature_mode=$feature_mode' \
  'host=$host' \
  'gpu=$gpu' \
  'epochs=$EPOCHS' \
  'seeds=$seeds_text' \
  'graphs=$graph_abs' \
  'upstream=$upstream' \
  'deps=$deps' > $run_info_q
set +e
$(quote_words "${train_cmd[@]}") 2>&1 | tee $stdout_q
status=\${PIPESTATUS[0]}
printf '%s\n' "\$status" > $exit_status_q
if [[ "\$status" -eq 0 ]]; then
  touch $output_q/COMPLETED
else
  touch $output_q/FAILED
fi
exit "\$status"
EOF
)

  repo_q="$(quote_one "$repo")"
  python_q="$(quote_one "$python_bin")"
  graph_q="$(quote_one "$graph_abs")"
  upstream_q="$(quote_one "$upstream")"
  deps_q="$(quote_one "$deps")"
  remote_script=$(cat <<EOF
set -euo pipefail
cd $repo_q
test -x $python_q
test -f $graph_q
test -d $upstream_q/.git
test -d $deps_q/GCL
if tmux has-session -t $(quote_one "$session_name") 2>/dev/null; then
  echo 'tmux session already exists: $session_name' >&2
  exit 21
fi
if [[ -e $output_q/COMPLETED || -e $output_q/FAILED ]]; then
  echo 'completed/failed output already exists: $output' >&2
  exit 22
fi
mkdir -p $output_q
tmux new-session -d -s $(quote_one "$session_name") \
  bash -lc $(quote_one "$session_cmd")
EOF
)

  echo
  echo "[graphcl] $host cuda:$gpu $dataset ($feature_mode)"
  echo "[graphcl] input: $graph_abs"
  echo "[graphcl] output: $output"
  ssh_cmd=(
    ssh -n "${SSH_OPTS[@]}" "$host"
    "bash -lc $(quote_one "$remote_script")"
  )
  if run_cmd "${ssh_cmd[@]}"; then
    launched=$((launched + 1))
    echo "[graphcl] session: $session_name"
  else
    failures=$((failures + 1))
  fi
done < "$SCHEDULE_FILE"

echo
echo "[summary] launched: $launched"
echo "[summary] failures: $failures"
exit "$failures"
