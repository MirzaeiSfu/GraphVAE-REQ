#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Launch one tmux-backed experiment per remote target over SSH.

Usage:
  scripts/launch_remote_tmux.sh \
    --repo-dir ~/GraphVAE-REQ \
    --env-activate 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate graphvae' \
    --config configs/kiarash_graphvae/qm9_graphvae.yaml \
    --experiment-name qm9_graphvae \
    --target ws1,cuda:0 \
    --target ws2,cuda:0 \
    --target ws2,cuda:1 \
    -- --plot_test_graphs false

Required launcher flags:
  --repo-dir PATH          Remote repository path.
  --config PATH            Config path relative to the repo, or absolute on the remote host.
  --experiment-name NAME   Short label used for tmux session and run directory names.
  --target HOST,DEVICE     Repeat once per remote run. HOST may be user@hostname.

Optional launcher flags:
  --env-activate CMD       Shell snippet to activate your environment on the remote host.
  --python-bin BIN         Python executable on the remote host. Default: python
  --run-root PATH          Root directory for per-run outputs inside the repo. Default: runs
  --git-pull               Run 'git pull --ff-only' on the remote repo before launching.
  --dry-run                Print the SSH commands without executing them.
  --help                   Show this message.

Any arguments after '--' are forwarded to main.py unchanged.
EOF
}

quote_words() {
  local quoted=()
  local word
  for word in "$@"; do
    quoted+=("$(printf '%q' "$word")")
  done
  printf '%s' "${quoted[*]}"
}

sanitize_token() {
  local value="$1"
  value="${value//@/_}"
  value="${value//./_}"
  value="${value//:/_}"
  value="${value//\//_}"
  value="${value//-/_}"
  value="${value// /_}"
  printf '%s' "$value"
}

REPO_DIR=""
ENV_ACTIVATE=""
CONFIG_PATH=""
EXPERIMENT_NAME=""
PYTHON_BIN="python"
RUN_ROOT="runs"
GIT_PULL=false
DRY_RUN=false
TARGET_SPECS=()
TRAIN_ARGS=()

while (($#)); do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --env-activate)
      ENV_ACTIVATE="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --experiment-name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --git-pull)
      GIT_PULL=true
      shift
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
    --target)
      TARGET_SPECS+=("$2")
      shift 2
      ;;
    *)
      echo "Unknown launcher argument: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$REPO_DIR" || -z "$CONFIG_PATH" || -z "$EXPERIMENT_NAME" ]]; then
  echo "--repo-dir, --config, and --experiment-name are required." >&2
  echo >&2
  usage >&2
  exit 1
fi

if ((${#TARGET_SPECS[@]} == 0)); then
  echo "Provide at least one --target HOST,DEVICE entry." >&2
  exit 1
fi

for target_spec in "${TARGET_SPECS[@]}"; do
  IFS=',' read -r host device extra <<<"$target_spec"

  if [[ -n "${extra:-}" || -z "${host:-}" || -z "${device:-}" ]]; then
    echo "Invalid --target '$target_spec'. Expected HOST,DEVICE." >&2
    exit 1
  fi

  host_token="$(sanitize_token "$host")"
  device_token="$(sanitize_token "$device")"
  session_name="${EXPERIMENT_NAME}_${host_token}_${device_token}"
  run_dir="${RUN_ROOT%/}/${EXPERIMENT_NAME}_${host_token}_${device_token}"

  train_cmd=(
    env
    MPLBACKEND=Agg
    PYTHONUNBUFFERED=1
    "$PYTHON_BIN"
    -u
    main.py
    --config "$CONFIG_PATH"
    --device "$device"
    --graph_save_path "$run_dir"
  )
  if ((${#TRAIN_ARGS[@]})); then
    train_cmd+=("${TRAIN_ARGS[@]}")
  fi
  train_cmd_str="$(quote_words "${train_cmd[@]}")"

  session_inner=""
  if [[ -n "$ENV_ACTIVATE" ]]; then
    session_inner+="$ENV_ACTIVATE && "
  fi
  session_inner+="cd $(printf '%q' "$REPO_DIR") && "
  session_inner+="mkdir -p $(printf '%q' "$run_dir") && "
  session_inner+="$train_cmd_str 2>&1 | tee $(printf '%q' "$run_dir/stdout.log")"

  remote_script="set -euo pipefail
if ! command -v tmux >/dev/null 2>&1; then
  echo 'tmux is not installed on this host.' >&2
  exit 1
fi
cd $(printf '%q' "$REPO_DIR")
"

  if [[ "$GIT_PULL" == true ]]; then
    remote_script+="git pull --ff-only
"
  fi

  remote_script+="if tmux has-session -t $(printf '%q' "$session_name") 2>/dev/null; then
  echo 'tmux session already exists: $session_name' >&2
  exit 1
fi
mkdir -p $(printf '%q' "$run_dir")
tmux new-session -d -s $(printf '%q' "$session_name") bash -lc $(printf '%q' "$session_inner")
"

  ssh_cmd=(ssh "$host" "bash -lc $(printf '%q' "$remote_script")")

  echo "[launch] $host -> session=$session_name device=$device run_dir=$run_dir"
  if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] $(quote_words "${ssh_cmd[@]}")"
  else
    "${ssh_cmd[@]}"
    echo "[attach] ssh $host -t 'tmux attach -t $session_name'"
    echo "[logs]   ssh $host -t 'tail -f $run_dir/stdout.log'"
  fi
  echo
done
