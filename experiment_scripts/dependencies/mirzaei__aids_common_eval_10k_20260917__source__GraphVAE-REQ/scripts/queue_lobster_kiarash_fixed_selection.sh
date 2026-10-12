#!/usr/bin/env bash
set -euo pipefail

if (($# < 2 || $# > 4)); then
  echo "Usage: $0 REPO_PATH HOST_LABEL [EXPECTED_RUNS] [GPU]" >&2
  exit 2
fi

repo_path="$1"
host_label="$2"
expected_runs="${3:-2}"
gpu="${4:-0}"
run_root="runs/20260724/lobster_kiarash_parity_fixed_split"
output_dir="runs/20260724/lobster_kiarash_parity_fixed_split_validation_selection/$host_label"
session="kiarash_fixed_select_20260724"
python_bin=""
for candidate in \
  /local-scratch2/mirzaei/miniconda3/envs/micro/bin/python \
  /localhome/mirzaei/miniconda3/envs/micro/bin/python
do
  if [[ -x "$candidate" ]]; then
    python_bin="$candidate"
    break
  fi
done
if [[ -z "$python_bin" ]]; then
  echo "Could not find the micro environment Python executable" >&2
  exit 3
fi

cd "$repo_path"
mkdir -p "$output_dir"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "already active: $session"
  exit 0
fi

worker_command=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "$repo_path")
while true; do
  completed=\$(find $(printf '%q' "$run_root") -name model_19999_0 | wc -l)
  if ((completed >= $expected_runs)); then
    break
  fi
  echo "[wait] completed=\$completed/$expected_runs \$(date -Iseconds)"
  sleep 30
done
env CUDA_VISIBLE_DEVICES=$(printf '%q' "$gpu") MPLBACKEND=Agg \
  $(printf '%q' "$python_bin") -u scripts/select_lobster_checkpoints_per_run.py \
  --runs-root $(printf '%q' "$run_root") \
  --output-dir $(printf '%q' "$output_dir") \
  --expected-runs $(printf '%q' "$expected_runs") \
  --expected-checkpoints-per-run 5 \
  --validation-rollouts 10 \
  --seed 20260724 \
  --device cuda:0 \
  --skip-materialization
EOF
)

tmux new-session -d -s "$session" bash -lc \
  "$worker_command 2>&1 | tee $(printf '%q' "$output_dir/stdout.log")"
echo "launched: $session"
